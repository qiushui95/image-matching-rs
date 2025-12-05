/*!
 * 图像匹配器统一接口
 *
 * 提供FFT和分段匹配两种模式的统一接口：
 * - FFT模式：基于J.P. Lewis的快速归一化互相关算法，适用于大图像的快速匹配
 * - 分段模式：基于分段归一化互相关算法，适用于内存受限或小图像的匹配
 *
 * 参考文献:
 * - J.P. Lewis: "Fast Normalized Cross-Correlation"
 *   http://scribblethink.org/Work/nvisionInterface/vi95_lewis.pdf
 */

use image::{DynamicImage, GenericImageView, ImageBuffer, Luma, imageops::FilterType};
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};
use std::cmp::{Ordering, max};
use std::error::Error;

/// 匹配器模式枚举
///
/// 定义了两种不同的图像匹配算法模式
#[derive(Debug, Clone, Copy)]
pub enum MatcherMode {
    /// FFT模式 - 使用快速傅里叶变换进行频域匹配
    ///
    /// 适用场景：
    /// - 大尺寸图像匹配
    /// - 需要高精度匹配
    /// - 内存充足的环境
    ///
    /// 参数：
    /// - width: 目标图像宽度
    /// - height: 目标图像高度
    FFT { width: u32, height: u32 },

    /// 分段模式 - 使用分段归一化互相关进行匹配
    ///
    /// 适用场景：
    /// - 小尺寸图像匹配
    /// - 内存受限环境
    /// - 需要快速粗略匹配
    Segmented,
}

#[derive(Debug, Clone)]
pub struct MatcherResult {
    /// 匹配区域宽度
    pub width: u32,
    /// 匹配区域高度
    pub height: u32,
    /// 最佳匹配结果
    pub best_result: MatcherSingleResult,
    /// 所有匹配结果
    pub all_result: Vec<MatcherSingleResult>,
}

impl MatcherResult {
    fn new(
        template_data: &TemplateData,
        all_result: Vec<MatcherSingleResult>,
    ) -> Result<Self, Box<dyn Error>> {
        if all_result.is_empty() {
            return Err("No matching results found".into());
        }

        let mut all_result = all_result;

        let best_result = all_result.remove(0);

        let result = Self {
            width: template_data.get_template_width(),
            height: template_data.get_template_height(),
            best_result,
            all_result,
        };

        Ok(result)
    }
}

/// 单个图像匹配结果
///
/// 包含匹配位置、尺寸和相关系数信息
#[derive(Debug, Clone)]
pub struct MatcherSingleResult {
    /// 匹配区域左上角X坐标
    pub x: u32,
    /// 匹配区域左上角Y坐标
    pub y: u32,
    /// 归一化互相关系数 (0.0-1.0，越接近1.0匹配度越高)
    pub correlation: f64,
}

impl PartialEq for MatcherSingleResult {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl Eq for MatcherSingleResult {}

impl PartialOrd for MatcherSingleResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.correlation.total_cmp(&other.correlation))
    }
}

impl Ord for MatcherSingleResult {
    fn cmp(&self, other: &Self) -> Ordering {
        self.correlation.total_cmp(&other.correlation)
    }
}

/// 匹配结果过滤器
///
/// 按坐标邻域对候选结果进行去重，以避免相近位置的重复结果。
pub struct MatcherResultFilter {
    x_delta: u32,
    y_delta: u32,
}
impl MatcherResultFilter {
    /// 创建过滤器
    ///
    /// # 参数
    /// * `x_delta` - X 方向允许的近邻范围
    /// * `y_delta` - Y 方向允许的近邻范围
    ///
    /// # 返回值
    /// 过滤器实例
    pub fn new(x_delta: u32, y_delta: u32) -> Self {
        Self { x_delta, y_delta }
    }

    /// 创建默认过滤器
    ///
    /// 默认近邻范围：`x_delta=5`, `y_delta=5`
    pub fn default() -> Self {
        Self::new(5, 5)
    }
    /// 判断候选是否需要根据近邻规则被过滤
    ///
    /// 当候选与已存在结果在 `x_delta/y_delta` 所定义的邻域内时，返回 `true`。
    fn need_filter(&self, item: &MatcherSingleResult, exist: &MatcherSingleResult) -> bool {
        if item.x < exist.x - self.x_delta {
            return false;
        }

        if item.x > exist.x + self.x_delta {
            return false;
        }

        if item.y < exist.y - self.y_delta {
            return false;
        }

        if item.y > exist.y + self.y_delta {
            return false;
        }

        true
    }
}

/// 模板数据枚举
///
/// 存储不同匹配模式下预处理的模板数据
#[derive(Debug, Clone)]
enum TemplateData {
    /// FFT模式的模板数据
    FFT { data: FFTTemplateData },
    /// 分段模式的模板数据
    Segmented { data: SegmentedTemplateData },
}

impl TemplateData {
    fn get_template_width(&self) -> u32 {
        match self {
            TemplateData::FFT { data } => data.template_width,
            TemplateData::Segmented { data } => data.template_width,
        }
    }

    fn get_template_height(&self) -> u32 {
        match self {
            TemplateData::FFT { data } => data.template_height,
            TemplateData::Segmented { data } => data.template_height,
        }
    }
}

/// FFT模式模板数据结构
///
/// 存储FFT匹配所需的预处理数据
#[derive(Debug, Clone)]
struct FFTTemplateData {
    /// 模板在频域的共轭复数数组
    template_conj_freq: Vec<Complex<f64>>,
    /// 模板的平方偏差和（用于归一化）
    template_sum_squared_deviations: f64,
    /// 模板宽度
    template_width: u32,
    /// 模板高度
    template_height: u32,
    /// FFT填充后的尺寸（2的幂次）
    padded_size: u32,
}

/// 分段模式模板数据结构
///
/// 存储分段匹配所需的预处理数据
#[derive(Debug, Clone)]
struct SegmentedTemplateData {
    /// 快速分段数据 - 格式: (x, y, width, height, mean_value)
    /// 用于初步筛选，分段数量较少，计算速度快
    template_segments_fast: Vec<(u32, u32, u32, u32, f64)>,
    /// 精细分段数据 - 格式: (x, y, width, height, mean_value)
    /// 用于精确匹配，分段数量较多，计算精度高
    template_segments_slow: Vec<(u32, u32, u32, u32, f64)>,
    /// 模板宽度
    template_width: u32,
    /// 模板高度
    template_height: u32,
    /// 快速分段的平方差之和
    segment_sum_squared_deviations_fast: f64,
    /// 精细分段的平方差之和
    segment_sum_squared_deviations_slow: f64,
    /// 快速分段的期望相关性
    expected_corr_fast: f64,
    /// 精细分段的期望相关性
    expected_corr_slow: f64,
    /// 快速分段的均值
    segments_mean_fast: f64,
    /// 精细分段的均值
    segments_mean_slow: f64,
}

/// 分段类型枚举
#[derive(Debug, Clone, Copy)]
enum SegmentType {
    Fast, // 快速分段
    Slow, // 精细分段
}

/// 调整后的阈值结构
struct AdjustedThresholds {
    fast_threshold: f64,
    slow_threshold: f64,
}

/// 图像统计信息结构
struct ImageStatistics {
    mean: f64,
    avg_deviation: f64,
}

/// 积分图像结构
struct IntegralImages {
    integral: Vec<Vec<u64>>,
    squared_integral: Vec<Vec<u64>>,
}

/// 图像匹配器主结构
///
/// 提供统一的图像匹配接口，支持FFT和分段两种匹配模式
pub struct ImageMatcher {
    /// 预处理的模板数据
    template_data: TemplateData,
    /// 匹配模式
    mode: MatcherMode,
    /// 模板图像宽度
    pub template_width: u32,
    /// 模板图像高度
    pub template_height: u32,
}

impl ImageMatcher {
    /// 从动态图像创建匹配器并预处理模板
    ///
    /// # 参数
    /// * `img` - 模板图像
    /// * `mode` - 匹配模式（FFT或分段）
    /// * `resize_width` - 可选的模板图像调整宽度
    ///
    /// # 返回值
    /// 返回配置好的ImageMatcher实例
    ///
    /// # 示例
    /// ```
    /// use image::open;
    /// use image_matching_rs::{ImageMatcher, MatcherMode};
    ///
    /// let template = open("template.png").unwrap();
    /// let matcher = ImageMatcher::new_from_image(
    ///     template,
    ///     MatcherMode::FFT { width: 1920, height: 1080 },
    ///     None
    /// );
    /// ```
    pub fn new_from_image(img: DynamicImage, mode: MatcherMode, resize_width: Option<u32>) -> Self {
        let img = Self::resize_image_if_needed(img, resize_width);
        let template_image = img.to_luma8();

        let (template_width, template_height) = template_image.dimensions();

        let template_data = match mode {
            MatcherMode::FFT { width, height } => {
                Self::prepare_fft_template_data(&template_image, width, height)
            }
            MatcherMode::Segmented => Self::prepare_segmented_template_data(&template_image),
        };

        Self {
            template_data,
            mode,
            template_width,
            template_height,
        }
    }

    /// 根据需要调整图像尺寸
    ///
    /// # 参数
    /// * `img` - 原始图像
    /// * `resize_width` - 可选的目标宽度
    ///
    /// # 返回值
    /// 调整后的图像（如果不需要调整则返回原图像）
    fn resize_image_if_needed(img: DynamicImage, resize_width: Option<u32>) -> DynamicImage {
        let Some(resize_width) = resize_width else {
            return img;
        };

        let (width, height) = img.dimensions();

        if width == resize_width {
            return img;
        }

        let resize_height = (height as f64 * resize_width as f64 / width as f64) as u32;
        img.resize(resize_width, resize_height, FilterType::Lanczos3)
    }

    /// 准备FFT模式的模板数据
    ///
    /// 将模板图像转换为频域表示，并计算相关统计信息
    ///
    /// # 参数
    /// * `template` - 模板图像
    /// * `image_width` - 目标图像宽度
    /// * `image_height` - 目标图像高度
    ///
    /// # 返回值
    /// 包含FFT数据的TemplateData
    fn prepare_fft_template_data(
        template: &ImageBuffer<Luma<u8>, Vec<u8>>,
        image_width: u32,
        image_height: u32,
    ) -> TemplateData {
        let (template_width, template_height) = template.dimensions();

        // 计算FFT所需的填充尺寸
        let required_width = image_width + template_width - 1;
        let required_height = image_height + template_height - 1;
        let padded_width = required_width.next_power_of_two();
        let padded_height = required_height.next_power_of_two();
        let padded_size = max(padded_width, padded_height);

        // 转换模板为二维数组
        let template_vec = Self::image_buffer_to_2d_vec(template);

        // 计算模板均值
        let template_mean = Self::calculate_image_mean(&template_vec);

        // 创建零均值模板
        let zero_mean_template = Self::create_zero_mean_template(&template_vec, template_mean);

        // 计算平方偏差和
        let template_sum_squared_deviations =
            Self::calculate_sum_squared_deviations(&zero_mean_template);

        // 创建填充的模板并进行FFT变换
        let template_conj_freq = Self::create_fft_template_conjugate(
            &zero_mean_template,
            template_width,
            template_height,
            padded_size,
        );

        let data = FFTTemplateData {
            template_conj_freq,
            template_sum_squared_deviations,
            template_width,
            template_height,
            padded_size,
        };

        TemplateData::FFT { data }
    }

    /// 准备分段模式的模板数据
    ///
    /// 将模板图像分割为多个分段，并计算每个分段的统计信息
    ///
    /// # 参数
    /// * `template` - 模板图像
    ///
    /// # 返回值
    /// 包含分段数据的TemplateData
    fn prepare_segmented_template_data(template: &ImageBuffer<Luma<u8>, Vec<u8>>) -> TemplateData {
        let (template_width, template_height) = template.dimensions();

        // 计算模板的基本统计信息
        let template_stats = Self::calculate_template_statistics(template);

        // 创建快速分段（粗糙分割）
        let (
            template_segments_fast,
            segment_sum_squared_deviations_fast,
            expected_corr_fast,
            segments_mean_fast,
        ) = Self::create_template_segments(
            template,
            template_stats.mean,
            template_stats.avg_deviation,
            SegmentType::Fast,
        );

        // 创建精细分段（详细分割）
        let (
            template_segments_slow,
            segment_sum_squared_deviations_slow,
            expected_corr_slow,
            segments_mean_slow,
        ) = Self::create_template_segments(
            template,
            template_stats.mean,
            template_stats.avg_deviation,
            SegmentType::Slow,
        );

        let data = SegmentedTemplateData {
            template_segments_fast,
            template_segments_slow,
            template_width,
            template_height,
            segment_sum_squared_deviations_fast,
            segment_sum_squared_deviations_slow,
            expected_corr_fast,
            expected_corr_slow,
            segments_mean_fast,
            segments_mean_slow,
        };

        TemplateData::Segmented { data }
    }

    /// 执行图像匹配
    ///
    /// # 参数
    /// * `img` - 待匹配的目标图像
    /// * `threshold` - 匹配阈值 (0.0-1.0)
    /// * `filter` - 可选的结果去重过滤器（按坐标邻域去重）
    ///
    /// # 返回值
    /// `MatcherResult`，包含最佳结果与全部结果（已按相关系数降序）
    ///
    /// # 错误
    /// 当图像尺寸不匹配或其他处理错误时返回错误
    ///
    /// # 示例
    /// ```
    /// let result = matcher.matching(target_image, 0.8, None)?;
    /// println!("最佳匹配: ({}, {}), 相关系数: {:.3}",
    ///          result.best_result.x, result.best_result.y, result.best_result.correlation);
    /// for r in &result.all_result {
    ///     println!("候选: ({}, {}), corr={:.3}", r.x, r.y, r.correlation);
    /// }
    /// ```
    pub fn matching(
        &self,
        img: DynamicImage,
        threshold: f64,
        filter: Option<MatcherResultFilter>,
    ) -> Result<MatcherResult, Box<dyn Error>> {
        let image = img.to_luma8();

        // 验证FFT模式下的图像尺寸
        if let MatcherMode::FFT { width, height } = self.mode {
            let (img_width, img_height) = img.dimensions();
            if width != img_width || height != img_height {
                let msg = format!(
                    "图像尺寸 {}x{} 与预设尺寸 {}x{} 不匹配",
                    img_width, img_height, width, height
                );

                return Err(msg.into());
            }
        }

        // 根据模板数据类型选择匹配算法
        let mut list = match &self.template_data {
            TemplateData::FFT { data } => Self::perform_fft_matching(data, image, threshold),
            TemplateData::Segmented { data } => {
                Self::perform_segmented_matching(data, image, threshold)
            }
        }?;

        list.sort_by(|a, b| a.cmp(b).reverse());

        let Some(filter) = filter else {
            return MatcherResult::new(&self.template_data, list);
        };

        let mut results = vec![];

        for item in list {
            let mut need_filter = false;

            for exist in results.iter() {
                if filter.need_filter(&item, exist) {
                    need_filter = true;
                    break;
                }
            }

            if need_filter {
                continue;
            }

            results.push(item);
        }

        MatcherResult::new(&self.template_data, results)
    }

    /// 执行FFT模式匹配
    ///
    /// 使用快速傅里叶变换在频域进行归一化互相关计算
    ///
    /// # 参数
    /// * `template_data` - FFT模板数据
    /// * `image` - 目标图像
    /// * `threshold` - 匹配阈值
    ///
    /// # 返回值
    /// 匹配结果列表
    fn perform_fft_matching(
        template_data: &FFTTemplateData,
        image: ImageBuffer<Luma<u8>, Vec<u8>>,
        threshold: f64,
    ) -> Result<Vec<MatcherSingleResult>, Box<dyn Error>> {
        let (image_width, image_height) = image.dimensions();

        // 检查图像尺寸
        if template_data.template_width > image_width {
            return Err("模板宽度大于图像宽度".into());
        }

        if template_data.template_height > image_height {
            return Err("模板高度大于图像高度".into());
        }

        // 计算积分图像用于快速区域统计
        let integral_images = Self::compute_integral_images(&image);

        // 创建零均值图像
        let zero_mean_image = Self::create_zero_mean_image(&image);

        // 执行FFT匹配计算
        let fft_result = Self::perform_fft_convolution(
            &zero_mean_image,
            image_width,
            image_height,
            template_data,
        );

        // 并行计算所有位置的相关系数
        Self::calculate_fft_correlations(
            &integral_images,
            template_data,
            &fft_result,
            image_width,
            image_height,
            threshold,
        )
    }

    /// 执行分段模式匹配
    ///
    /// 使用分段归一化互相关算法进行匹配
    ///
    /// # 参数
    /// * `template_data` - 分段模板数据
    /// * `image` - 目标图像
    /// * `threshold` - 匹配阈值
    ///
    /// # 返回值
    /// 匹配结果列表
    fn perform_segmented_matching(
        template_data: &SegmentedTemplateData,
        image: ImageBuffer<Luma<u8>, Vec<u8>>,
        threshold: f64,
    ) -> Result<Vec<MatcherSingleResult>, Box<dyn Error>> {
        let (template_width, template_height) =
            (template_data.template_width, template_data.template_height);
        let (image_width, image_height) = image.dimensions();

        // 检查图像尺寸
        if template_width > image_width {
            return Err("模板宽度大于图像宽度".into());
        }

        if template_height > image_height {
            return Err("模板高度大于图像高度".into());
        }

        // 计算积分图像
        let integral_images = Self::compute_integral_images(&image);

        // 计算调整后的期望相关性阈值
        let adjusted_thresholds = Self::calculate_adjusted_thresholds(template_data, threshold);

        // 并行计算所有位置的相关系数
        Self::calculate_segmented_correlations(
            &integral_images,
            template_data,
            &adjusted_thresholds,
            image_width,
            image_height,
            template_width,
            template_height,
        )
    }

    // ==================== 辅助函数 ====================

    /// 将ImageBuffer转换为二维向量
    ///
    /// # 参数
    /// * `image` - 输入图像
    ///
    /// # 返回值
    /// 二维像素值数组
    fn image_buffer_to_2d_vec(image: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Vec<Vec<u8>> {
        let (width, height) = image.dimensions();
        let mut result = vec![vec![0u8; width as usize]; height as usize];

        for y in 0..height {
            for x in 0..width {
                result[y as usize][x as usize] = image.get_pixel(x, y)[0];
            }
        }

        result
    }

    /// 计算图像统计信息
    ///
    /// # 参数
    /// * `image_vec` - 二维像素数组
    ///
    /// # 返回值
    /// 包含均值的统计信息
    fn calculate_image_mean(image_vec: &[Vec<u8>]) -> f64 {
        let height = image_vec.len();
        let width = image_vec[0].len();
        let total_pixels = (height * width) as f64;

        let sum: u64 = image_vec
            .iter()
            .flat_map(|row| row.iter())
            .map(|&val| val as u64)
            .sum();

        sum as f64 / total_pixels
    }

    /// 计算模板统计信息（包括平均偏差）
    ///
    /// # 参数
    /// * `template` - 模板图像
    ///
    /// # 返回值
    /// 包含均值和平均偏差的统计信息
    fn calculate_template_statistics(template: &ImageBuffer<Luma<u8>, Vec<u8>>) -> ImageStatistics {
        let (width, height) = template.dimensions();
        let total_pixels = (width * height) as f64;

        // 计算均值
        let sum: u64 = (0..height)
            .flat_map(|y| (0..width).map(move |x| template.get_pixel(x, y)[0] as u64))
            .sum();
        let mean = sum as f64 / total_pixels;

        // 计算平均偏差
        let sum_deviations: f64 = (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let pixel_value = template.get_pixel(x, y)[0] as f64;
                    (pixel_value - mean).abs()
                })
            })
            .sum();
        let avg_deviation = sum_deviations / total_pixels;

        ImageStatistics {
            mean,
            avg_deviation,
        }
    }

    /// 创建零均值模板
    ///
    /// # 参数
    /// * `template_vec` - 原始模板数据
    /// * `mean` - 模板均值
    ///
    /// # 返回值
    /// 零均值模板数据
    fn create_zero_mean_template(template_vec: &[Vec<u8>], mean: f64) -> Vec<Vec<f64>> {
        template_vec
            .iter()
            .map(|row| row.iter().map(|&pixel| pixel as f64 - mean).collect())
            .collect()
    }

    /// 计算平方偏差和
    ///
    /// # 参数
    /// * `zero_mean_template` - 零均值模板
    ///
    /// # 返回值
    /// 平方偏差和
    fn calculate_sum_squared_deviations(zero_mean_template: &[Vec<f64>]) -> f64 {
        zero_mean_template
            .iter()
            .flat_map(|row| row.iter())
            .map(|&val| val * val)
            .sum()
    }

    /// 创建FFT模板的频域共轭
    ///
    /// # 参数
    /// * `zero_mean_template` - 零均值模板
    /// * `template_width` - 模板宽度
    /// * `template_height` - 模板高度
    /// * `padded_size` - 填充尺寸
    ///
    /// # 返回值
    /// 频域共轭复数数组
    fn create_fft_template_conjugate(
        zero_mean_template: &[Vec<f64>],
        template_width: u32,
        template_height: u32,
        padded_size: u32,
    ) -> Vec<Complex<f64>> {
        // 创建填充的模板复数向量
        let mut template_padded =
            vec![Complex::new(0.0, 0.0); (padded_size * padded_size) as usize];

        for y in 0..template_height {
            for x in 0..template_width {
                let pixel_value = zero_mean_template[y as usize][x as usize];
                template_padded[y as usize * padded_size as usize + x as usize] =
                    Complex::new(pixel_value, 0.0);
            }
        }

        // 执行FFT变换
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward((padded_size * padded_size) as usize);
        fft.process(&mut template_padded);

        // 返回共轭
        template_padded.iter().map(|&val| val.conj()).collect()
    }

    /// 创建零均值图像
    ///
    /// # 参数
    /// * `image` - 输入图像
    ///
    /// # 返回值
    /// 零均值图像数据
    fn create_zero_mean_image(image: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Vec<Vec<f64>> {
        let (width, height) = image.dimensions();

        // 计算图像均值
        let sum: u64 = (0..height)
            .flat_map(|y| (0..width).map(move |x| image.get_pixel(x, y)[0] as u64))
            .sum();
        let mean = sum as f64 / (width * height) as f64;

        // 创建零均值图像
        (0..height)
            .map(|y| {
                (0..width)
                    .map(|x| image.get_pixel(x, y)[0] as f64 - mean)
                    .collect()
            })
            .collect()
    }

    /// 执行FFT卷积
    ///
    /// # 参数
    /// * `zero_mean_image` - 零均值图像
    /// * `image_width` - 图像宽度
    /// * `image_height` - 图像高度
    /// * `template_data` - FFT模板数据
    ///
    /// # 返回值
    /// FFT卷积结果
    fn perform_fft_convolution(
        zero_mean_image: &[Vec<f64>],
        image_width: u32,
        image_height: u32,
        template_data: &FFTTemplateData,
    ) -> Vec<Complex<f64>> {
        // 填充图像到指定尺寸
        let mut image_padded = vec![
            Complex::new(0.0, 0.0);
            (template_data.padded_size * template_data.padded_size) as usize
        ];

        for y in 0..image_height {
            for x in 0..image_width {
                let pixel_value = zero_mean_image[y as usize][x as usize];
                image_padded[y as usize * template_data.padded_size as usize + x as usize] =
                    Complex::new(pixel_value, 0.0);
            }
        }

        // FFT变换图像到频域
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner
            .plan_fft_forward((template_data.padded_size * template_data.padded_size) as usize);
        fft.process(&mut image_padded);

        // 频域相乘（互相关）
        let product_freq: Vec<Complex<f64>> = image_padded
            .iter()
            .zip(template_data.template_conj_freq.iter())
            .map(|(&img_val, &tmpl_val)| img_val * tmpl_val)
            .collect();

        // 逆FFT
        let mut fft_result = product_freq;
        let ifft = planner
            .plan_fft_inverse((template_data.padded_size * template_data.padded_size) as usize);
        ifft.process(&mut fft_result);

        fft_result
    }

    /// 计算积分图像
    ///
    /// 积分图像用于快速计算任意矩形区域的像素和
    ///
    /// # 参数
    /// * `image` - 输入图像
    ///
    /// # 返回值
    /// 包含普通积分图像和平方积分图像的结构
    fn compute_integral_images(image: &ImageBuffer<Luma<u8>, Vec<u8>>) -> IntegralImages {
        let (width, height) = image.dimensions();
        let mut integral = vec![vec![0u64; width as usize + 1]; height as usize + 1];
        let mut squared_integral = vec![vec![0u64; width as usize + 1]; height as usize + 1];

        for y in 0..height {
            for x in 0..width {
                let pixel_value = image.get_pixel(x, y)[0] as u64;
                let squared_value = pixel_value * pixel_value;

                integral[y as usize + 1][x as usize + 1] = pixel_value
                    + integral[y as usize][x as usize + 1]
                    + integral[y as usize + 1][x as usize]
                    - integral[y as usize][x as usize];

                squared_integral[y as usize + 1][x as usize + 1] = squared_value
                    + squared_integral[y as usize][x as usize + 1]
                    + squared_integral[y as usize + 1][x as usize]
                    - squared_integral[y as usize][x as usize];
            }
        }

        IntegralImages {
            integral,
            squared_integral,
        }
    }

    /// 使用积分图像快速计算矩形区域的像素和
    ///
    /// # 参数
    /// * `integral` - 积分图像
    /// * `x` - 区域左上角X坐标
    /// * `y` - 区域左上角Y坐标
    /// * `width` - 区域宽度
    /// * `height` - 区域高度
    ///
    /// # 返回值
    /// 区域内像素值的和
    fn sum_region(integral: &[Vec<u64>], x: u32, y: u32, width: u32, height: u32) -> u64 {
        let x1 = x as usize;
        let y1 = y as usize;
        let x2 = (x + width) as usize;
        let y2 = (y + height) as usize;

        integral[y2][x2] + integral[y1][x1] - integral[y1][x2] - integral[y2][x1]
    }

    /// 计算FFT相关系数
    ///
    /// # 参数
    /// * `integral_images` - 积分图像
    /// * `template_data` - FFT模板数据
    /// * `fft_result` - FFT卷积结果
    /// * `image_width` - 图像宽度
    /// * `image_height` - 图像高度
    /// * `threshold` - 匹配阈值
    ///
    /// # 返回值
    /// 匹配结果列表
    fn calculate_fft_correlations(
        integral_images: &IntegralImages,
        template_data: &FFTTemplateData,
        fft_result: &[Complex<f64>],
        image_width: u32,
        image_height: u32,
        threshold: f64,
    ) -> Result<Vec<MatcherSingleResult>, Box<dyn Error>> {
        // 生成所有坐标对，用于并行处理
        let coords: Vec<(u32, u32)> = (0..=(image_height - template_data.template_height))
            .flat_map(|y| (0..=(image_width - template_data.template_width)).map(move |x| (x, y)))
            .collect();

        // 并行计算所有可能位置的相关系数
        let list: Vec<MatcherSingleResult> = coords
            .par_iter()
            .map(|&(x, y)| {
                let correlation = Self::calculate_single_fft_correlation(
                    &integral_images.integral,
                    &integral_images.squared_integral,
                    template_data,
                    x,
                    y,
                    fft_result,
                );
                (x, y, correlation)
            })
            .filter(|&(_, _, correlation)| correlation >= threshold)
            .map(|(x, y, correlation)| MatcherSingleResult { x, y, correlation })
            .collect();

        Ok(list)
    }

    /// 计算单个位置的FFT相关系数
    ///
    /// # 参数
    /// * `image_integral` - 图像积分图像
    /// * `squared_image_integral` - 平方积分图像
    /// * `template_data` - FFT模板数据
    /// * `x` - X坐标
    /// * `y` - Y坐标
    /// * `fft_result` - FFT结果
    ///
    /// # 返回值
    /// 归一化互相关系数
    fn calculate_single_fft_correlation(
        image_integral: &[Vec<u64>],
        squared_image_integral: &[Vec<u64>],
        template_data: &FFTTemplateData,
        x: u32,
        y: u32,
        fft_result: &[Complex<f64>],
    ) -> f64 {
        // 计算分子（从FFT结果获取，需要归一化）
        let numerator = fft_result[y as usize * template_data.padded_size as usize + x as usize].re
            / (template_data.padded_size * template_data.padded_size) as f64;

        // 计算分母
        let sum_image_region = Self::sum_region(
            image_integral,
            x,
            y,
            template_data.template_width,
            template_data.template_height,
        ) as f64;

        let sum_squared_image_region = Self::sum_region(
            squared_image_integral,
            x,
            y,
            template_data.template_width,
            template_data.template_height,
        ) as f64;

        let template_size = (template_data.template_width * template_data.template_height) as f64;
        let image_mean_squared = (sum_image_region * sum_image_region) / template_size;
        let image_sum_squared_deviations = sum_squared_image_region - image_mean_squared;

        let denominator = (template_data.template_sum_squared_deviations
            * image_sum_squared_deviations)
            .sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            let correlation = numerator / denominator;
            // 限制相关系数在合理范围内
            if correlation > 2.0 || correlation < -2.0 {
                0.0
            } else {
                correlation
            }
        }
    }

    /// 计算调整后的阈值
    ///
    /// # 参数
    /// * `template_data` - 分段模板数据
    /// * `threshold` - 原始阈值
    ///
    /// # 返回值
    /// 调整后的阈值结构
    fn calculate_adjusted_thresholds(
        template_data: &SegmentedTemplateData,
        threshold: f64,
    ) -> AdjustedThresholds {
        // 分段相关性通常低于像素级互相关，适当降低阈值上限以提高召回
        let base = threshold.min(0.95);
        AdjustedThresholds {
            fast_threshold: base * template_data.expected_corr_fast - 0.0001,
            slow_threshold: base * template_data.expected_corr_slow - 0.0001,
        }
    }

    /// 计算分段相关系数
    ///
    /// # 参数
    /// * `integral_images` - 积分图像
    /// * `template_data` - 分段模板数据
    /// * `thresholds` - 调整后的阈值
    /// * `image_width` - 图像宽度
    /// * `image_height` - 图像高度
    /// * `template_width` - 模板宽度
    /// * `template_height` - 模板高度
    ///
    /// # 返回值
    /// 匹配结果列表
    fn calculate_segmented_correlations(
        integral_images: &IntegralImages,
        template_data: &SegmentedTemplateData,
        thresholds: &AdjustedThresholds,
        image_width: u32,
        image_height: u32,
        template_width: u32,
        template_height: u32,
    ) -> Result<Vec<MatcherSingleResult>, Box<dyn Error>> {
        // 生成所有可能的坐标
        let coords: Vec<(u32, u32)> = (0..=(image_height - template_height))
            .flat_map(|y| (0..=(image_width - template_width)).map(move |x| (x, y)))
            .collect();

        // 使用并行计算进行匹配
        let list: Vec<MatcherSingleResult> = coords
            .par_iter()
            .map(|&(x, y)| {
                let correlation = Self::calculate_single_segmented_correlation(
                    &integral_images.integral,
                    &integral_images.squared_integral,
                    template_data,
                    x,
                    y,
                    thresholds.fast_threshold,
                );
                (x, y, correlation)
            })
            .filter(|&(_, _, correlation)| correlation >= thresholds.slow_threshold)
            .map(|(x, y, correlation)| MatcherSingleResult { x, y, correlation })
            .collect();

        Ok(list)
    }

    /// 计算单个位置的分段相关系数
    ///
    /// 使用两级分段匹配：先用快速分段筛选，再用精细分段验证
    ///
    /// # 参数
    /// * `image_integral` - 图像积分图像
    /// * `squared_image_integral` - 平方积分图像
    /// * `template_data` - 分段模板数据
    /// * `x` - X坐标
    /// * `y` - Y坐标
    /// * `fast_threshold` - 快速分段阈值
    ///
    /// # 返回值
    /// 归一化互相关系数
    fn calculate_single_segmented_correlation(
        image_integral: &[Vec<u64>],
        squared_image_integral: &[Vec<u64>],
        template_data: &SegmentedTemplateData,
        x: u32,
        y: u32,
        fast_threshold: f64,
    ) -> f64 {
        // 第一阶段：快速分段匹配
        let fast_corr = Self::calculate_segment_correlation(
            image_integral,
            squared_image_integral,
            &template_data.template_segments_fast,
            template_data.template_width,
            template_data.template_height,
            template_data.segment_sum_squared_deviations_fast,
            template_data.segments_mean_fast,
            x,
            y,
        );

        // 如果快速匹配不通过，直接返回
        if fast_corr < fast_threshold {
            return fast_corr;
        }

        // 第二阶段：精细分段匹配
        Self::calculate_segment_correlation(
            image_integral,
            squared_image_integral,
            &template_data.template_segments_slow,
            template_data.template_width,
            template_data.template_height,
            template_data.segment_sum_squared_deviations_slow,
            template_data.segments_mean_slow,
            x,
            y,
        )
    }

    /// 计算分段相关系数的核心逻辑
    ///
    /// # 参数
    /// * `image_integral` - 图像积分图像
    /// * `squared_image_integral` - 平方积分图像
    /// * `segments` - 分段数据
    /// * `template_width` - 模板宽度
    /// * `template_height` - 模板高度
    /// * `segment_sum_squared_deviations` - 分段平方偏差和
    /// * `segments_mean` - 分段均值
    /// * `x` - X坐标
    /// * `y` - Y坐标
    ///
    /// # 返回值
    /// 相关系数
    fn calculate_segment_correlation(
        image_integral: &[Vec<u64>],
        squared_image_integral: &[Vec<u64>],
        segments: &[(u32, u32, u32, u32, f64)],
        template_width: u32,
        template_height: u32,
        segment_sum_squared_deviations: f64,
        segments_mean: f64,
        x: u32,
        y: u32,
    ) -> f64 {
        let mut numerator = 0.0f64;

        // 计算整个模板区域的图像统计信息
        let total_image_sum =
            Self::sum_region(image_integral, x, y, template_width, template_height) as f64;
        let template_size = (template_width * template_height) as f64;
        let image_mean = total_image_sum / template_size;

        // 计算图像分段均值的方差（基于全局均值），用于与模板分段均值匹配的归一化
        let mut image_segment_sum_squared_deviations = 0.0f64;

        // 遍历所有分段计算相关性分子
        for &(seg_x, seg_y, seg_width, seg_height, segment_mean) in segments {
            let region_sum =
                Self::sum_region(image_integral, x + seg_x, y + seg_y, seg_width, seg_height)
                    as f64;

            let region_size = (seg_width * seg_height) as f64;
            let region_mean = region_sum / region_size;

            // 计算分子项（模板分段均值 与 图像分段均值 的协方差加权）
            numerator += (segment_mean - segments_mean) * (region_mean - image_mean) * region_size;

            // 图像分段均值的平方偏差和（与分母中的模板分段平方偏差和同尺度）
            image_segment_sum_squared_deviations +=
                (region_mean - image_mean) * (region_mean - image_mean) * region_size;
        }

        // 计算最终相关系数
        let denominator =
            (segment_sum_squared_deviations * image_segment_sum_squared_deviations).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            let corr = numerator / denominator;
            // 数值安全处理：限制相关系数在 [-1.0, 1.0] 范围内
            corr.clamp(-1.0, 1.0)
        }
    }

    /// 创建模板分段
    ///
    /// 使用递归分治算法将模板图像分割为多个分段
    ///
    /// # 参数
    /// * `template` - 模板图像
    /// * `mean_template_value` - 模板均值
    /// * `avg_deviation_of_template` - 模板平均偏差
    /// * `segment_type` - 分段类型（快速或精细）
    ///
    /// # 返回值
    /// (分段列表, 平方偏差和, 期望相关性, 分段均值)
    fn create_template_segments(
        template: &ImageBuffer<Luma<u8>, Vec<u8>>,
        mean_template_value: f64,
        avg_deviation_of_template: f64,
        segment_type: SegmentType,
    ) -> (Vec<(u32, u32, u32, u32, f64)>, f64, f64, f64) {
        let (template_width, template_height) = template.dimensions();
        let mut picture_segments: Vec<(u32, u32, u32, u32, f64)> = Vec::new();

        // 根据分段类型设置不同的参数
        let (max_segments, min_std_dev_multiplier) = match segment_type {
            SegmentType::Fast => (25, 0.8),  // 快速分段：较少分段，较大标准差阈值
            SegmentType::Slow => (100, 0.6), // 精细分段：较多分段，较小标准差阈值
        };

        let mut segments_mean = 0.0;
        let mut min_std_dev = avg_deviation_of_template * min_std_dev_multiplier;

        // 循环调整阈值直到分段数量合适
        while picture_segments.len() > max_segments || picture_segments.is_empty() {
            picture_segments.clear();
            segments_mean = 0.0;

            // 调用递归分段函数
            Self::recursive_binary_segmentation(
                template,
                0,
                0,
                template_width,
                template_height,
                min_std_dev,
                &mut picture_segments,
            );

            // 计算分段均值
            if !picture_segments.is_empty() {
                let sum_means: f64 = picture_segments.iter().map(|(_, _, _, _, mean)| mean).sum();
                segments_mean = sum_means / picture_segments.len() as f64;
            }

            // 如果分段过多，增加标准差阈值
            if picture_segments.len() > max_segments {
                min_std_dev *= 1.1;
            } else if picture_segments.is_empty() {
                min_std_dev *= 0.9;
            }

            // 防止无限循环
            if min_std_dev > avg_deviation_of_template * 2.0 || min_std_dev < 0.1 {
                break;
            }
        }

        // 如果仍然没有分段，创建一个包含整个模板的分段
        if picture_segments.is_empty() {
            picture_segments.push((0, 0, template_width, template_height, mean_template_value));
            segments_mean = mean_template_value;
        }

        // 合并相邻的相似分段
        let merged_segments = Self::merge_similar_segments(picture_segments);

        // 计算统计信息
        let (segment_sum_squared_deviations, expected_corr) =
            Self::calculate_segment_statistics(&merged_segments, segments_mean);

        (
            merged_segments,
            segment_sum_squared_deviations,
            expected_corr,
            segments_mean,
        )
    }

    /// 递归二进制分段算法
    ///
    /// 基于标准差阈值递归分割图像区域
    ///
    /// # 参数
    /// * `template` - 模板图像
    /// * `x` - 区域左上角X坐标
    /// * `y` - 区域左上角Y坐标
    /// * `width` - 区域宽度
    /// * `height` - 区域高度
    /// * `min_std_dev` - 最小标准差阈值
    /// * `segments` - 分段结果列表
    fn recursive_binary_segmentation(
        template: &ImageBuffer<Luma<u8>, Vec<u8>>,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        min_std_dev: f64,
        segments: &mut Vec<(u32, u32, u32, u32, f64)>,
    ) {
        // 计算当前区域的统计信息
        let (mean, std_dev) = Self::calculate_region_statistics(template, x, y, width, height);

        // 如果标准差小于阈值或区域太小，停止分割
        if std_dev < min_std_dev || width < 4 || height < 4 {
            segments.push((x, y, width, height, mean));
            return;
        }

        // 选择分割方向（优先分割较长的边）
        if width >= height {
            // 水平分割
            let mid_x = width / 2;
            Self::recursive_binary_segmentation(
                template,
                x,
                y,
                mid_x,
                height,
                min_std_dev,
                segments,
            );
            Self::recursive_binary_segmentation(
                template,
                x + mid_x,
                y,
                width - mid_x,
                height,
                min_std_dev,
                segments,
            );
        } else {
            // 垂直分割
            let mid_y = height / 2;
            Self::recursive_binary_segmentation(
                template,
                x,
                y,
                width,
                mid_y,
                min_std_dev,
                segments,
            );
            Self::recursive_binary_segmentation(
                template,
                x,
                y + mid_y,
                width,
                height - mid_y,
                min_std_dev,
                segments,
            );
        }
    }

    /// 计算图像区域的统计信息
    ///
    /// # 参数
    /// * `template` - 模板图像
    /// * `x` - 区域左上角X坐标
    /// * `y` - 区域左上角Y坐标
    /// * `width` - 区域宽度
    /// * `height` - 区域高度
    ///
    /// # 返回值
    /// (均值, 标准差)
    fn calculate_region_statistics(
        template: &ImageBuffer<Luma<u8>, Vec<u8>>,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> (f64, f64) {
        let mut sum = 0u64;
        let mut sum_squared = 0u64;
        let pixel_count = (width * height) as f64;

        for dy in 0..height {
            for dx in 0..width {
                let pixel_value = template.get_pixel(x + dx, y + dy)[0] as u64;
                sum += pixel_value;
                sum_squared += pixel_value * pixel_value;
            }
        }

        let mean = sum as f64 / pixel_count;
        let variance = (sum_squared as f64 / pixel_count) - (mean * mean);
        let std_dev = variance.sqrt();

        (mean, std_dev)
    }

    /// 合并相似的相邻分段
    ///
    /// # 参数
    /// * `segments` - 原始分段列表
    ///
    /// # 返回值
    /// 合并后的分段列表
    fn merge_similar_segments(
        mut segments: Vec<(u32, u32, u32, u32, f64)>,
    ) -> Vec<(u32, u32, u32, u32, f64)> {
        let mut changed = true;

        while changed {
            changed = false;
            let mut i = 0;

            while i < segments.len() {
                let mut j = i + 1;

                while j < segments.len() {
                    if Self::should_merge_segments(&segments[i], &segments[j]) {
                        let merged = Self::merge_two_segments(&segments[i], &segments[j]);
                        segments[i] = merged;
                        segments.remove(j);
                        changed = true;
                    } else {
                        j += 1;
                    }
                }
                i += 1;
            }
        }

        segments
    }

    /// 判断两个分段是否应该合并
    ///
    /// # 参数
    /// * `seg1` - 第一个分段
    /// * `seg2` - 第二个分段
    ///
    /// # 返回值
    /// 是否应该合并
    fn should_merge_segments(
        seg1: &(u32, u32, u32, u32, f64),
        seg2: &(u32, u32, u32, u32, f64),
    ) -> bool {
        let (x1, y1, w1, h1, mean1) = *seg1;
        let (x2, y2, w2, h2, mean2) = *seg2;

        // 检查是否相邻
        let adjacent = (x1 + w1 == x2 && y1 == y2 && h1 == h2) ||  // 水平相邻
            (x1 == x2 && y1 + h1 == y2 && w1 == w2) ||  // 垂直相邻
            (x2 + w2 == x1 && y1 == y2 && h1 == h2) ||  // 水平相邻（反向）
            (x1 == x2 && y2 + h2 == y1 && w1 == w2); // 垂直相邻（反向）

        // 检查均值是否相似（差异小于10%）
        let mean_similar = (mean1 - mean2).abs() < (mean1 + mean2) * 0.05;

        adjacent && mean_similar
    }

    /// 合并两个分段
    ///
    /// # 参数
    /// * `seg1` - 第一个分段
    /// * `seg2` - 第二个分段
    ///
    /// # 返回值
    /// 合并后的分段
    fn merge_two_segments(
        seg1: &(u32, u32, u32, u32, f64),
        seg2: &(u32, u32, u32, u32, f64),
    ) -> (u32, u32, u32, u32, f64) {
        let (x1, y1, w1, h1, mean1) = *seg1;
        let (x2, y2, w2, h2, mean2) = *seg2;

        let min_x = x1.min(x2);
        let min_y = y1.min(y2);
        let max_x = (x1 + w1).max(x2 + w2);
        let max_y = (y1 + h1).max(y2 + h2);

        let new_width = max_x - min_x;
        let new_height = max_y - min_y;

        // 按面积加权计算新的均值
        let area1 = (w1 * h1) as f64;
        let area2 = (w2 * h2) as f64;
        let total_area = area1 + area2;
        let new_mean = (mean1 * area1 + mean2 * area2) / total_area;

        (min_x, min_y, new_width, new_height, new_mean)
    }

    /// 计算分段统计信息
    ///
    /// # 参数
    /// * `segments` - 分段列表
    /// * `segments_mean` - 分段均值
    ///
    /// # 返回值
    /// (平方偏差和, 期望相关性)
    fn calculate_segment_statistics(
        segments: &[(u32, u32, u32, u32, f64)],
        segments_mean: f64,
    ) -> (f64, f64) {
        let mut sum_squared_deviations = 0.0f64;
        let mut total_area = 0u32;

        for &(_, _, width, height, segment_mean) in segments {
            let area = width * height;
            let deviation = segment_mean - segments_mean;
            sum_squared_deviations += deviation * deviation * area as f64;
            total_area += area;
        }

        // 计算期望相关性（基于分段的方差）
        let variance = sum_squared_deviations / total_area as f64;
        let expected_corr = (variance / (variance + 1.0)).sqrt();

        (sum_squared_deviations, expected_corr)
    }
}
