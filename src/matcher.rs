/*!
 * 图像匹配器统一接口
 *
 * 提供FFT和分段匹配两种模式的统一接口
 * FFT模式基于J.P. Lewis的快速归一化互相关算法
 * 论文: "Fast Normalized Cross-Correlation"
 * http://scribblethink.org/Work/nvisionInterface/vi95_lewis.pdf
 */

use image::{imageops::FilterType, DynamicImage, ImageBuffer, Luma};
use rayon::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};
use std::cmp::max;

/// 匹配器模式
#[derive(Debug, Clone, Copy)]
pub enum MatcherMode {
    /// FFT模式 - 适用于大图像的快速匹配
    FFT,
    /// 分段模式 - 适用于内存受限或小图像的匹配
    Segmented,
}

/// 匹配结果
#[derive(Debug, Clone)]
pub struct MatcherResult {
    /// X坐标
    pub x: u32,
    /// Y坐标
    pub y: u32,
    /// 匹配区域宽度
    pub width: u32,
    /// 匹配区域高度
    pub height: u32,
    /// 相关系数 (0.0-1.0)
    pub correlation: f32,
}

/// FFT模板数据结构
#[derive(Debug, Clone)]
struct FFTTemplateData {
    /// 模板在频域的共轭
    template_conj_freq: Vec<Complex<f32>>,
    /// 模板的平方偏差和
    template_sum_squared_deviations: f32,
    /// 模板宽度
    template_width: u32,
    /// 模板高度
    template_height: u32,
    /// 填充后的尺寸
    padded_size: u32,
}

/// 图像匹配器
///
/// 提供统一的图像匹配接口，支持FFT和分段两种匹配模式
pub struct ImageMatcher {
    /// FFT模板数据
    fft_template_data: Option<FFTTemplateData>,
    /// 分段模板数据
    segmented_template_data: Option<SegmentedTemplateData>,
    /// 原始模板图像（用于分段匹配）
    template_image: Option<ImageBuffer<Luma<u8>, Vec<u8>>>,
    /// 模板尺寸
    template_size: Option<(u32, u32)>,
}

impl ImageMatcher {
    /// 创建新的图像匹配器
    pub fn new() -> Self {
        Self {
            fft_template_data: None,
            segmented_template_data: None,
            template_image: None,
            template_size: None,
        }
    }

    /// 从动态图像创建匹配器并准备模板
    ///
    /// # 参数
    /// * `img` - 模板图像
    /// * `resize_width` - 可选的调整宽度
    pub fn from_image(img: DynamicImage, resize_width: Option<u32>) -> Self {
        let img = Self::resize_image(img, resize_width);
        let template = img.to_luma8();

        let mut matcher = Self::new();
        matcher.template_image = Some(template);
        matcher.template_size = Some(matcher.template_image.as_ref().unwrap().dimensions());
        matcher
    }

    /// 调整图像尺寸
    fn resize_image(img: DynamicImage, resize_width: Option<u32>) -> DynamicImage {
        let Some(resize_width) = resize_width else {
            return img;
        };

        let resize_height = (img.height() as f32 * resize_width as f32 / img.width() as f32) as u32;
        img.resize(resize_width, resize_height, FilterType::Lanczos3)
    }

    /// 准备模板用于匹配
    ///
    /// # 参数
    /// * `template` - 模板图像
    /// * `image_width` - 目标图像宽度
    /// * `image_height` - 目标图像高度
    /// * `mode` - 匹配模式
    pub fn prepare_template(
        &mut self,
        template: &ImageBuffer<Luma<u8>, Vec<u8>>,
        image_width: u32,
        image_height: u32,
        mode: MatcherMode,
    ) -> Result<(), String> {
        self.template_image = Some(template.clone());
        self.template_size = Some(template.dimensions());

        match mode {
            MatcherMode::FFT => {
                self.fft_template_data = Some(self.prepare_template_fft(template, image_width, image_height));
                Ok(())
            }
            MatcherMode::Segmented => {
                // 准备分段模板数据
                let segmented_data = self.prepare_segmented_template(template)?;
                self.segmented_template_data = Some(segmented_data);
                Ok(())
            }
        }
    }

    /// 执行图像匹配
    ///
    /// # 参数
    /// * `img` - 输入图像
    /// * `mode` - 匹配模式
    /// * `threshold` - 匹配阈值
    pub fn matching(
        &self,
        img: DynamicImage,
        mode: MatcherMode,
        threshold: f32,
    ) -> Result<Vec<MatcherResult>, String> {
        let image = img.to_luma8();

        match mode {
            MatcherMode::FFT => self.match_by_fft(&image, threshold),
            MatcherMode::Segmented => self.match_by_segmented(&image, threshold),
        }
    }

    /// 使用FFT模式进行匹配
    fn match_by_fft(&self, image: &ImageBuffer<Luma<u8>, Vec<u8>>, threshold: f32) -> Result<Vec<MatcherResult>, String> {
        let template_data = self.fft_template_data.as_ref()
            .ok_or("FFT模板未预处理，请先调用prepare_template")?;

        let matches = self.fft_template_match(image, threshold, template_data)?;

        // 转换为MatcherResult格式
        let results = matches.into_iter().map(|(x, y, correlation)| {
            MatcherResult {
                x,
                y,
                width: template_data.template_width,
                height: template_data.template_height,
                correlation: correlation as f32,
            }
        }).collect();

        Ok(results)
    }

    /// 使用分段模式进行匹配 - 基于rustautogui的fast_ncc_template_match算法
    pub fn match_by_segmented(&self, image: &ImageBuffer<Luma<u8>, Vec<u8>>, threshold: f32) -> Result<Vec<MatcherResult>, String> {
        // 获取预处理的分段模板数据
        let template_data = self.segmented_template_data.as_ref()
            .ok_or("分段模板数据未设置，请先调用 prepare_template")?;

        let (template_width, template_height) = (template_data.template_width, template_data.template_height);
        let (image_width, image_height) = image.dimensions();

        if template_width > image_width || template_height > image_height {
            return Ok(Vec::new());
        }

        // 计算积分图像
        let (image_integral, squared_image_integral) = self.compute_integral_images(image);

        // 计算调整后的期望相关性
        let adjusted_fast_expected_corr: f32 = threshold * template_data.expected_corr_fast - 0.0001;
        let adjusted_slow_expected_corr: f32 = threshold * template_data.expected_corr_slow - 0.0001;

        // 生成所有可能的坐标
        let coords: Vec<(u32, u32)> = (0..=(image_height - template_height))
            .flat_map(|y| (0..=(image_width - template_width)).map(move |x| (x, y)))
            .collect();

        // 使用并行计算进行匹配
        use rayon::prelude::*;
        let mut found_points: Vec<(u32, u32, f32)> = coords
            .par_iter()
            .map(|&(x, y)| {
                let corr = self.fast_correlation_calculation(
                    &image_integral,
                    &squared_image_integral,
                    &template_data.template_segments_fast,
                    &template_data.template_segments_slow,
                    template_width,
                    template_height,
                    template_data.segment_sum_squared_deviations_fast,
                    template_data.segment_sum_squared_deviations_slow,
                    template_data.segments_mean_fast,
                    template_data.segments_mean_slow,
                    x,
                    y,
                    adjusted_fast_expected_corr,
                );
                (x, y, corr as f32)
            })
            .filter(|&(_, _, corr)| corr >= adjusted_slow_expected_corr)
            .collect();

        // 按相关系数降序排序
        found_points.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // 转换为MatcherResult格式
        let results = found_points.into_iter().map(|(x, y, correlation)| {
            MatcherResult {
                x,
                y,
                width: template_width,
                height: template_height,
                correlation,
            }
        }).collect();

        Ok(results)
    }

    /// 获取模板尺寸
    pub fn template_size(&self) -> Option<(u32, u32)> {
        self.template_size
    }

    // ==================== FFT 实现部分 ====================

    /// 预处理模板图像，生成FFT数据
    fn prepare_template_fft(
        &self,
        template: &ImageBuffer<Luma<u8>, Vec<u8>>,
        image_width: u32,
        image_height: u32,
    ) -> FFTTemplateData {
        let (template_width, template_height) = template.dimensions();

        // 使用更小的填充大小：只需要足够进行卷积的大小
        let required_width = image_width + template_width - 1;
        let required_height = image_height + template_height - 1;
        let padded_width = required_width.next_power_of_two();
        let padded_height = required_height.next_power_of_two();
        let padded_size = max(padded_width, padded_height);

        // 转换模板为二维向量
        let template_vec = self.imagebuffer_to_vec(template);

        // 计算模板的平均值
        let template_sum: u64 = template_vec.iter()
            .flat_map(|row| row.iter())
            .map(|&val| val as u64)
            .sum();
        let template_average = template_sum as f32 / (template_width * template_height) as f32;

        // 计算零均值模板和平方偏差和
        let mut template_sum_squared_deviations = 0.0f32;
        let mut zero_mean_template = vec![vec![0.0f32; template_width as usize]; template_height as usize];

        for y in 0..template_height {
            for x in 0..template_width {
                let pixel_value = template_vec[y as usize][x as usize] as f32;
                let zero_mean_value = pixel_value - template_average;
                zero_mean_template[y as usize][x as usize] = zero_mean_value;
                template_sum_squared_deviations += zero_mean_value * zero_mean_value;
            }
        }

        // 创建填充的模板复数向量
        let mut template_padded = vec![Complex::new(0.0, 0.0); (padded_size * padded_size) as usize];
        for y in 0..template_height {
            for x in 0..template_width {
                let pixel_value = zero_mean_template[y as usize][x as usize];
                template_padded[y as usize * padded_size as usize + x as usize] =
                    Complex::new(pixel_value, 0.0);
            }
        }

        // 执行FFT变换
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward((padded_size * padded_size) as usize);
        fft.process(&mut template_padded);

        // 计算共轭
        let template_conj_freq: Vec<Complex<f32>> = template_padded
            .iter()
            .map(|&val| val.conj())
            .collect();

        FFTTemplateData {
            template_conj_freq,
            template_sum_squared_deviations,
            template_width,
            template_height,
            padded_size,
        }
    }

    /// 执行FFT模板匹配
    fn fft_template_match(
        &self,
        image: &ImageBuffer<Luma<u8>, Vec<u8>>,
        threshold: f32,
        template_data: &FFTTemplateData,
    ) -> Result<Vec<(u32, u32, f64)>, String> {
        let (image_width, image_height) = image.dimensions();

        // 检查图像尺寸
        if image_width < template_data.template_width || image_height < template_data.template_height {
            return Ok(Vec::new());
        }

        // 计算积分图像
        let (image_integral, squared_image_integral) = self.compute_integral_images(image);

        // 计算图像平均值
        let sum_image: u64 = self.sum_region(&image_integral, 0, 0, image_width, image_height);
        let image_average = sum_image as f32 / (image_height * image_width) as f32;

        // 创建零均值图像
        let mut zero_mean_image = vec![vec![0.0f32; image_width as usize]; image_height as usize];
        for y in 0..image_height {
            for x in 0..image_width {
                let pixel_value = image.get_pixel(x, y)[0] as f32;
                zero_mean_image[y as usize][x as usize] = pixel_value - image_average;
            }
        }

        // 填充图像到指定尺寸
        let mut image_padded = vec![Complex::new(0.0, 0.0); (template_data.padded_size * template_data.padded_size) as usize];
        for y in 0..image_height {
            for x in 0..image_width {
                let pixel_value = zero_mean_image[y as usize][x as usize];
                image_padded[y as usize * template_data.padded_size as usize + x as usize] =
                    Complex::new(pixel_value, 0.0);
            }
        }

        // FFT变换图像到频域
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward((template_data.padded_size * template_data.padded_size) as usize);
        fft.process(&mut image_padded);

        // 频域相乘（互相关）
        let product_freq: Vec<Complex<f32>> = image_padded
            .iter()
            .zip(template_data.template_conj_freq.iter())
            .map(|(&img_val, &tmpl_val)| img_val * tmpl_val)
            .collect();

        // 逆FFT
        let mut fft_result = product_freq;
        let ifft = planner.plan_fft_inverse((template_data.padded_size * template_data.padded_size) as usize);
        ifft.process(&mut fft_result);

        // 生成所有坐标对，用于并行处理
        let coords: Vec<(u32, u32)> = (0..=(image_height - template_data.template_height))
            .flat_map(|y| (0..=(image_width - template_data.template_width)).map(move |x| (x, y)))
            .collect();

        // 并行计算所有可能位置的相关系数
        let found_points: Vec<(u32, u32, f64)> = coords
            .par_iter()
            .map(|&(x, y)| {
                let correlation = self.fft_correlation_calculation(
                    &image_integral,
                    &squared_image_integral,
                    template_data.template_width,
                    template_data.template_height,
                    template_data.template_sum_squared_deviations,
                    x,
                    y,
                    template_data.padded_size,
                    &fft_result,
                );
                (x, y, correlation)
            })
            .filter(|&(_, _, corr)| corr >= threshold as f64)
            .collect();

        // 按相关系数降序排序
        let mut sorted_points = found_points;
        sorted_points.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        Ok(sorted_points)
    }

    /// 计算FFT相关系数
    fn fft_correlation_calculation(
        &self,
        image_integral: &[Vec<u64>],
        squared_image_integral: &[Vec<u64>],
        template_width: u32,
        template_height: u32,
        template_sum_squared_deviations: f32,
        x: u32,
        y: u32,
        padded_size: u32,
        fft_result: &[Complex<f32>],
    ) -> f64 {
        // 计算分子（从FFT结果获取，需要归一化）
        let numerator = fft_result[y as usize * padded_size as usize + x as usize].re as f64
            / (padded_size * padded_size) as f64;

        // 计算分母
        let sum_image_region = self.sum_region(image_integral, x, y, template_width, template_height) as f64;
        let sum_squared_image_region = self.sum_region(squared_image_integral, x, y, template_width, template_height) as f64;

        let template_size = (template_width * template_height) as f64;
        let image_mean_squared = (sum_image_region * sum_image_region) / template_size;
        let image_sum_squared_deviations = sum_squared_image_region - image_mean_squared;

        let denominator = (template_sum_squared_deviations as f64 * image_sum_squared_deviations).sqrt();

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

    /// 将ImageBuffer转换为二维向量
    fn imagebuffer_to_vec(&self, image: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Vec<Vec<u8>> {
        let (width, height) = image.dimensions();
        let mut result = vec![vec![0u8; width as usize]; height as usize];

        for y in 0..height {
            for x in 0..width {
                result[y as usize][x as usize] = image.get_pixel(x, y)[0];
            }
        }

        result
    }

    /// 计算积分图像
    fn compute_integral_images(&self, image: &ImageBuffer<Luma<u8>, Vec<u8>>) -> (Vec<Vec<u64>>, Vec<Vec<u64>>) {
        let (width, height) = image.dimensions();
        let mut integral = vec![vec![0u64; width as usize + 1]; height as usize + 1];
        let mut squared_integral = vec![vec![0u64; width as usize + 1]; height as usize + 1];

        for y in 0..height {
            for x in 0..width {
                let pixel_value = image.get_pixel(x, y)[0] as u64;
                let squared_value = pixel_value * pixel_value;

                integral[y as usize + 1][x as usize + 1] =
                    pixel_value +
                    integral[y as usize][x as usize + 1] +
                    integral[y as usize + 1][x as usize] -
                    integral[y as usize][x as usize];

                squared_integral[y as usize + 1][x as usize + 1] =
                    squared_value +
                    squared_integral[y as usize][x as usize + 1] +
                    squared_integral[y as usize + 1][x as usize] -
                    squared_integral[y as usize][x as usize];
            }
        }

        (integral, squared_integral)
    }

    /// 准备分段模板数据 - 使用rustautogui的算法
    fn prepare_segmented_template(&self, template: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Result<SegmentedTemplateData, String> {
        // 调用rustautogui的prepare_template_picture算法
        self.prepare_template_picture(template, false, None)
    }

    /// 预处理模板图像 - 移植自rustautogui
    ///
    /// 预处理所有图像子图像
    /// 返回：
    /// - template_segments_fast: 用于低精度高速度的最少分段数的分段图像
    /// - template_segments_slow: 用于高精度低速度的高分段数的分段图像
    /// - template_width, template_height: 模板尺寸
    /// - segment_sum_squared_deviations_fast/slow: 用于分母计算的平方差之和
    /// - expected_corr_fast/slow: 分段模板与原始模板之间的相关性，用于确定最小期望相关性
    /// - segments_mean_fast/slow: 分段模板图像的平均值
    ///
    /// 图像基于其平均标准差进行分段，使用二进制分段
    /// 每个分段代表平均标准差低于某个阈值的区域
    /// 创建2个分段图像：快速的具有非常高的阈值，意味着同一区域内像素之间的偏差更高
    /// 慢速的具有高精度，意味着低偏差和更多分段
    /// 每个分段中的所有像素都设置为其均值的值
    fn prepare_template_picture(
        &self,
        template: &ImageBuffer<Luma<u8>, Vec<u8>>,
        debug: bool,
        corr_threshold: Option<f32>,
    ) -> Result<SegmentedTemplateData, String> {
        let (template_width, template_height) = template.dimensions();

        // 计算模板的均值和平均偏差
        let mut sum_template = 0u64;
        for y in 0..template_height {
            for x in 0..template_width {
                sum_template += template.get_pixel(x, y)[0] as u64;
            }
        }
        let mean_template_value = sum_template as f32 / (template_width * template_height) as f32;

        // 计算平均偏差
        let mut sum_deviations = 0.0f32;
        for y in 0..template_height {
            for x in 0..template_width {
                let pixel_value = template.get_pixel(x, y)[0] as f32;
                sum_deviations += (pixel_value - mean_template_value).abs();
            }
        }
        let avg_deviation_of_template = sum_deviations / (template_width * template_height) as f32;

        // 创建快速分段
        let (template_segments_fast, segment_sum_squared_deviations_fast, expected_corr_fast, segments_mean_fast) =
            self.create_picture_segments(template, mean_template_value, avg_deviation_of_template, "fast", corr_threshold)?;

        // 创建慢速分段
        let (template_segments_slow, segment_sum_squared_deviations_slow, expected_corr_slow, segments_mean_slow) =
            self.create_picture_segments(template, mean_template_value, avg_deviation_of_template, "slow", corr_threshold)?;

        if debug {
            // 保存调试图像（可选实现）
            println!("Debug mode: Fast segments: {}, Slow segments: {}",
                     template_segments_fast.len(), template_segments_slow.len());
        }

        Ok(SegmentedTemplateData {
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
        })
    }

    /// 创建图像分段 - 移植自rustautogui
    ///
    /// 返回 (picture_segments, segment_sum_squared_deviations, expected_corr, segments_mean)
    /// 调用递归分治二进制分段函数，该函数基于最小标准差阈值分割图像
    ///
    /// 如果创建了太多分段，阈值会在循环中增加，直到满足条件
    fn create_picture_segments(
        &self,
        template: &ImageBuffer<Luma<u8>, Vec<u8>>,
        mean_template_value: f32,
        avg_deviation_of_template: f32,
        template_type: &str,
        corr_threshold: Option<f32>,
    ) -> Result<(Vec<(u32, u32, u32, u32, f32)>, f32, f32, f32), String> {
        let (template_width, template_height) = template.dimensions();
        let mut picture_segments: Vec<(u32, u32, u32, u32, f32)> = Vec::new();

        // 调用递归函数将图像分割为相似像素值的分段
        let mut target_corr = 0.0;
        let mut threshold = 0.0;
        let mut v2_active = false;

        if template_type == "fast" {
            match corr_threshold {
                Some(x) => {
                    target_corr = x.min(0.85);
                    v2_active = true;
                }
                None => target_corr = -0.9,
            }
            threshold = 0.99;
        } else if template_type == "slow" {
            threshold = 0.85;
            target_corr = 0.99;
        }

        let mut expected_corr = -1.0;
        let mut segments_sum = 0;
        let mut segment_sum_squared_deviations = 0.0;

        // 循环直到满足条件
        while expected_corr < target_corr && segments_sum < 500 {
            picture_segments.clear();

            // 调用分治算法
            self.divide_and_conquer(&mut picture_segments, template, 0, 0, threshold);

            // 合并相邻分段
            let merged_segments = self.merge_picture_segments(picture_segments.clone());
            picture_segments = merged_segments;

            segments_sum = picture_segments.len();

            // 计算分段的统计信息
            let mut segments_mean = 0.0;
            segment_sum_squared_deviations = 0.0;

            for &(x, y, width, height, segment_mean) in &picture_segments {
                let segment_size = (width * height) as f32;
                segments_mean += segment_mean * segment_size;

                // 计算该分段的平方差
                let deviation = segment_mean - mean_template_value;
                segment_sum_squared_deviations += deviation * deviation * segment_size;
            }

            segments_mean /= (template_width * template_height) as f32;

            // 计算期望相关性
            if template_type == "slow" {
                // 计算分段模板与原始模板的相关性
                expected_corr = self.calculate_template_correlation(template, &picture_segments, mean_template_value);
            } else {
                // 对于快速模板，使用不同的计算方式
                if v2_active {
                    expected_corr = self.calculate_template_correlation(template, &picture_segments, mean_template_value);
                } else {
                    let current_distance = avg_deviation_of_template / (segments_sum as f32).sqrt();
                    expected_corr = if current_distance < 10.0 { 1.0 } else { -1.0 };
                }
            }

            // 调整阈值
            threshold += 0.05;
            if threshold > 2.0 {
                break;
            }
        }

        Ok((picture_segments, segment_sum_squared_deviations, expected_corr, mean_template_value))
    }

    /// 分治算法 - 递归分割图像
    ///
    /// 递归地将模板图像分割为相似颜色的区域，基于平均标准差与阈值的比较
    fn divide_and_conquer(
        &self,
        picture_segments: &mut Vec<(u32, u32, u32, u32, f32)>,
        segment: &ImageBuffer<Luma<u8>, Vec<u8>>,
        x: u32,
        y: u32,
        threshold: f32,
    ) {
        let (segment_width, segment_height) = segment.dimensions();

        if segment_width == 0 || segment_height == 0 {
            return;
        }

        // 计算分段的均值
        let mut sum = 0u64;
        let pixel_count = segment_width * segment_height;

        for seg_y in 0..segment_height {
            for seg_x in 0..segment_width {
                sum += segment.get_pixel(seg_x, seg_y)[0] as u64;
            }
        }

        let segment_mean = sum as f32 / pixel_count as f32;

        // 计算平方差之和
        let mut sum_squared_deviations = 0.0f32;
        for seg_y in 0..segment_height {
            for seg_x in 0..segment_width {
                let pixel_value = segment.get_pixel(seg_x, seg_y)[0] as f32;
                let deviation = pixel_value - segment_mean;
                sum_squared_deviations += deviation * deviation;
            }
        }

        // 计算平均偏差
        let average_deviation = (sum_squared_deviations / pixel_count as f32).sqrt();

        if average_deviation > threshold {
            // 分割图像
            if segment_width >= segment_height || segment_height == 1 {
                // 如果图像宽度大于等于高度，水平分割
                let mut additional_pixel = 0;
                if segment_width % 2 == 1 {
                    additional_pixel = 1;
                }

                let left_width = segment_width / 2 + additional_pixel;
                let right_width = segment_width / 2;

                // 创建左半部分
                let left_image = self.cut_screen_region(0, 0, left_width, segment_height, segment);
                // 创建右半部分
                let right_image = self.cut_screen_region(left_width, 0, right_width, segment_height, segment);

                let x1 = x + left_width;
                // 递归处理左右两部分
                self.divide_and_conquer(picture_segments, &left_image, x, y, threshold);
                self.divide_and_conquer(picture_segments, &right_image, x1, y, threshold);
            } else {
                // 如果图像高度大于宽度，垂直分割
                let mut additional_pixel = 0;
                if segment_height % 2 == 1 {
                    additional_pixel = 1;
                }

                let top_height = segment_height / 2 + additional_pixel;
                let bottom_height = segment_height / 2;

                // 创建上半部分
                let top_image = self.cut_screen_region(0, 0, segment_width, top_height, segment);
                // 创建下半部分
                let bottom_image = self.cut_screen_region(0, top_height, segment_width, bottom_height, segment);

                let y1 = y + top_height;
                // 递归处理上下两部分
                self.divide_and_conquer(picture_segments, &top_image, x, y, threshold);
                self.divide_and_conquer(picture_segments, &bottom_image, x, y1, threshold);
            }
        } else {
            // 递归退出 - 添加分段信息
            let segment_info = (x, y, segment_width, segment_height, segment_mean);
            picture_segments.push(segment_info);
        }
    }

    /// 裁剪屏幕区域 - 移植自rustautogui的cut_screen_region
    fn cut_screen_region(
        &self,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        screen_image: &ImageBuffer<Luma<u8>, Vec<u8>>,
    ) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        assert!(x + width <= screen_image.width());
        assert!(y + height <= screen_image.height());

        let mut sub_image: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::new(width, height);

        // 从原始图像缓冲区复制像素到子图像缓冲区
        for y_sub in 0..height {
            for x_sub in 0..width {
                let pixel = screen_image.get_pixel(x + x_sub, y + y_sub);
                sub_image.put_pixel(x_sub, y_sub, *pixel);
            }
        }
        sub_image
    }

    /// 合并图像分段 - 移植自rustautogui
    fn merge_picture_segments(&self, mut segmented_template: Vec<(u32, u32, u32, u32, f32)>) -> Vec<(u32, u32, u32, u32, f32)> {
        // 按x坐标然后按y坐标排序
        segmented_template.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        let mut changed = true;
        while changed {
            changed = false;
            let mut new_segments = Vec::new();
            let mut merged_indices = std::collections::HashSet::new();

            for i in 0..segmented_template.len() {
                if merged_indices.contains(&i) {
                    continue;
                }

                let mut current_segment = segmented_template[i];
                let mut was_merged = false;

                // 尝试垂直合并
                for j in (i + 1)..segmented_template.len() {
                    if merged_indices.contains(&j) {
                        continue;
                    }

                    let other_segment = segmented_template[j];

                    // 检查是否可以垂直合并
                    if current_segment.0 == other_segment.0 && // 相同x坐标
                       current_segment.2 == other_segment.2 && // 相同宽度
                       current_segment.4 == other_segment.4 && // 相同均值
                       current_segment.1 + current_segment.3 == other_segment.1 { // 垂直相邻

                        // 合并分段
                        current_segment.3 += other_segment.3; // 增加高度
                        merged_indices.insert(j);
                        was_merged = true;
                        changed = true;
                    }
                }

                new_segments.push(current_segment);
            }

            segmented_template = new_segments;
        }

        // 水平合并
        segmented_template.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

        let mut changed = true;
        while changed {
            changed = false;
            let mut new_segments = Vec::new();
            let mut merged_indices = std::collections::HashSet::new();

            for i in 0..segmented_template.len() {
                if merged_indices.contains(&i) {
                    continue;
                }

                let mut current_segment = segmented_template[i];

                // 尝试水平合并
                for j in (i + 1)..segmented_template.len() {
                    if merged_indices.contains(&j) {
                        continue;
                    }

                    let other_segment = segmented_template[j];

                    // 检查是否可以水平合并
                    if current_segment.1 == other_segment.1 && // 相同y坐标
                       current_segment.3 == other_segment.3 && // 相同高度
                       current_segment.4 == other_segment.4 && // 相同均值
                       current_segment.0 + current_segment.2 == other_segment.0 { // 水平相邻

                        // 合并分段
                        current_segment.2 += other_segment.2; // 增加宽度
                        merged_indices.insert(j);
                        changed = true;
                    }
                }

                new_segments.push(current_segment);
            }

            segmented_template = new_segments;
        }

        // 最终排序
        segmented_template.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        segmented_template
    }

    /// 计算模板相关性
    fn calculate_template_correlation(
        &self,
        template: &ImageBuffer<Luma<u8>, Vec<u8>>,
        segments: &[(u32, u32, u32, u32, f32)],
        template_mean: f32,
    ) -> f32 {
        let (template_width, template_height) = template.dimensions();
        let mut numerator = 0.0f32;
        let mut template_variance = 0.0f32;
        let mut segment_variance = 0.0f32;

        // 计算分段图像的均值
        let mut segment_mean = 0.0f32;
        for &(_, _, width, height, value) in segments {
            segment_mean += value * (width * height) as f32;
        }
        segment_mean /= (template_width * template_height) as f32;

        // 计算相关性
        for y in 0..template_height {
            for x in 0..template_width {
                let template_value = template.get_pixel(x, y)[0] as f32;

                // 找到对应的分段值
                let mut segment_value = template_mean; // 默认值
                for &(seg_x, seg_y, seg_width, seg_height, seg_value) in segments {
                    if x >= seg_x && x < seg_x + seg_width && y >= seg_y && y < seg_y + seg_height {
                        segment_value = seg_value;
                        break;
                    }
                }

                let template_diff = template_value - template_mean;
                let segment_diff = segment_value - segment_mean;

                numerator += template_diff * segment_diff;
                template_variance += template_diff * template_diff;
                segment_variance += segment_diff * segment_diff;
            }
        }

        let denominator = (template_variance * segment_variance).sqrt();
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// 快速相关性计算 - 移植自rustautogui
    ///
    /// 两阶段相关性计算：
    /// 1. 使用快速分段进行初步筛选
    /// 2. 对通过初步筛选的位置使用慢速分段进行精确计算
    fn fast_correlation_calculation(
        &self,
        image_integral: &[Vec<u64>],
        squared_image_integral: &[Vec<u64>],
        template_segments_fast: &[(u32, u32, u32, u32, f32)], // 粗略分段，分段数少
        template_segments_slow: &[(u32, u32, u32, u32, f32)], // 精确分段，分段数多
        template_width: u32,
        template_height: u32,
        fast_segments_sum_squared_deviations: f32,
        slow_segments_sum_squared_deviations: f32,
        segments_fast_mean: f32,
        segments_slow_mean: f32,
        x: u32, // 大图像x值
        y: u32, // 大图像y值
        min_expected_corr: f32,
    ) -> f64 {
        let template_area = template_height * template_width;

        /////////// 分子计算
        let sum_image: u64 = self.sum_region(image_integral, x, y, template_width, template_height);
        let mean_image = sum_image as f32 / (template_height * template_width) as f32;
        let mut nominator = 0.0;

        for (x1, y1, segment_width, segment_height, segment_value) in template_segments_fast {
            let segment_image_sum = self.sum_region(
                image_integral,
                x + x1,
                y + y1,
                *segment_width,
                *segment_height,
            );
            let segment_nominator_value: f32 = (segment_image_sum as f32
                - mean_image * (segment_height * segment_width) as f32)
                * (*segment_value - segments_fast_mean);
            nominator += segment_nominator_value;
        }

        ////////// 分母计算
        let sum_squared_image: u64 = self.sum_region(
            squared_image_integral,
            x,
            y,
            template_width,
            template_height,
        );
        let image_sum_squared_deviations =
            sum_squared_image as f32 - (sum_image as f32).powi(2) / template_area as f32;
        let denominator = (image_sum_squared_deviations * fast_segments_sum_squared_deviations).sqrt();
        let mut corr: f32 = nominator / denominator;

        ///////////////

        if corr > 1.1 || corr.is_nan() {
            corr = -100.0;
            return corr as f64;
        }

        // 使用更详细图像进行第二次计算
        if corr >= min_expected_corr {
            nominator = 0.0;
            for (x1, y1, segment_width, segment_height, segment_value) in template_segments_slow {
                let segment_image_sum = self.sum_region(
                    image_integral,
                    x + x1,
                    y + y1,
                    *segment_width,
                    *segment_height,
                );
                let segment_nominator_value: f32 = (segment_image_sum as f32
                    - mean_image * (segment_height * segment_width) as f32)
                    * (*segment_value - segments_slow_mean);
                nominator += segment_nominator_value;
            }

            let denominator =
                (image_sum_squared_deviations * slow_segments_sum_squared_deviations).sqrt();

            corr = nominator / denominator;
        }

        if corr > 1.1 || corr.is_nan() {
            corr = -100.0;
            return corr as f64;
        }

        corr as f64
    }



    /// 使用积分图像计算区域和
    fn sum_region(&self, integral: &[Vec<u64>], x: u32, y: u32, width: u32, height: u32) -> u64 {
        let x1 = x as usize;
        let y1 = y as usize;
        let x2 = (x + width) as usize;
        let y2 = (y + height) as usize;

        integral[y2][x2] + integral[y1][x1] - integral[y1][x2] - integral[y2][x1]
    }

}

impl Default for ImageMatcher {
    fn default() -> Self {
        Self::new()
    }
}



/// 分段模板数据 - 与rustautogui的SegmentedData结构兼容
#[derive(Debug, Clone)]
struct SegmentedTemplateData {
    /// 快速分段（粗糙分割，段数少）- 格式: (x, y, width, height, mean_value)
    template_segments_fast: Vec<(u32, u32, u32, u32, f32)>,
    /// 慢速分段（精细分割，段数多）- 格式: (x, y, width, height, mean_value)
    template_segments_slow: Vec<(u32, u32, u32, u32, f32)>,
    /// 模板宽度
    template_width: u32,
    /// 模板高度
    template_height: u32,
    /// 快速分段的平方差之和
    segment_sum_squared_deviations_fast: f32,
    /// 慢速分段的平方差之和
    segment_sum_squared_deviations_slow: f32,
    /// 快速分段的期望相关性
    expected_corr_fast: f32,
    /// 慢速分段的期望相关性
    expected_corr_slow: f32,
    /// 快速分段的均值
    segments_mean_fast: f32,
    /// 慢速分段的均值
    segments_mean_slow: f32,
}
