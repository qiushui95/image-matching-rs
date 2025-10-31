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

    /// 使用分段模式进行匹配
    pub fn match_by_segmented(&self, image: &ImageBuffer<Luma<u8>, Vec<u8>>, threshold: f32) -> Result<Vec<MatcherResult>, String> {
        // 获取预处理的分段模板数据
        let template_data = self.segmented_template_data.as_ref()
            .ok_or("分段模板数据未设置，请先调用 prepare_template")?;
        
        let (template_width, template_height) = (template_data.template_width, template_data.template_height);
        let (image_width, image_height) = image.dimensions();
        
        if template_width > image_width || template_height > image_height {
            return Ok(Vec::new());
        }
        
        // 预处理：创建积分图像
        let (image_integral, squared_image_integral) = self.compute_integral_images(image);
        
        // 执行分段匹配
        let mut results = Vec::new();
        
        for y in 0..=(image_height - template_height) {
            for x in 0..=(image_width - template_width) {
                let correlation = self.calculate_segmented_correlation(
                    &image_integral,
                    &squared_image_integral,
                    template_data,
                    x,
                    y,
                    template_width,
                    template_height,
                );
                
                if correlation >= threshold {
                    results.push(MatcherResult {
                        x,
                        y,
                        width: template_width,
                        height: template_height,
                        correlation,
                    });
                }
            }
        }
        
        // 按相关性降序排序
        results.sort_by(|a, b| b.correlation.partial_cmp(&a.correlation).unwrap_or(std::cmp::Ordering::Equal));
        
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

    /// 准备分段模板数据
    fn prepare_segmented_template(&self, template: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Result<SegmentedTemplateData, String> {
        let (template_width, template_height) = template.dimensions();
        
        // 计算模板的基本统计信息
        let template_sum: u64 = (0..template_height)
            .flat_map(|y| (0..template_width).map(move |x| template.get_pixel(x, y)[0] as u64))
            .sum();
        let template_mean = template_sum as f32 / (template_width * template_height) as f32;
        
        // 根据模板大小选择合适的分段策略
        let (fast_grid_x, fast_grid_y, slow_grid_x, slow_grid_y) = if template_width < 20 || template_height < 20 {
            // 小模板使用较少的分段
            (2, 2, 4, 4)
        } else if template_width < 50 || template_height < 50 {
            // 中等模板使用中等分段
            (4, 4, 8, 8)
        } else {
            // 大模板使用更多分段
            (8, 8, 16, 16)
        };
        
        // 创建快速分段
        let fast_segments = self.create_segments(template, fast_grid_x, fast_grid_y, template_mean);
        let fast_sum_squared_deviations = self.calculate_segments_variance(&fast_segments, template_mean);
        
        // 创建慢速分段
        let slow_segments = self.create_segments(template, slow_grid_x, slow_grid_y, template_mean);
        let slow_sum_squared_deviations = self.calculate_segments_variance(&slow_segments, template_mean);
        
        Ok(SegmentedTemplateData {
            fast_segments,
            slow_segments,
            template_width,
            template_height,
            fast_sum_squared_deviations,
            slow_sum_squared_deviations,
            fast_mean: template_mean,
            slow_mean: template_mean,
            expected_corr_fast: 0.1,  // 降低快速检查阈值
        })
    }
    
    /// 创建图像分段
    fn create_segments(&self, template: &ImageBuffer<Luma<u8>, Vec<u8>>, grid_x: u32, grid_y: u32, template_mean: f32) -> Vec<Segment> {
        let (template_width, template_height) = template.dimensions();
        let mut segments = Vec::new();
        
        let segment_width = template_width / grid_x;
        let segment_height = template_height / grid_y;
        
        for grid_row in 0..grid_y {
            for grid_col in 0..grid_x {
                let x = grid_col * segment_width;
                let y = grid_row * segment_height;
                
                let actual_width = if grid_col == grid_x - 1 {
                    template_width - x
                } else {
                    segment_width
                };
                
                let actual_height = if grid_row == grid_y - 1 {
                    template_height - y
                } else {
                    segment_height
                };
                
                // 计算段的平均值
                let mut sum = 0u64;
                let mut count = 0u32;
                
                for seg_y in y..(y + actual_height) {
                    for seg_x in x..(x + actual_width) {
                        sum += template.get_pixel(seg_x, seg_y)[0] as u64;
                        count += 1;
                    }
                }
                
                let value = if count > 0 { sum as f32 / count as f32 } else { template_mean };
                
                segments.push(Segment {
                    x,
                    y,
                    width: actual_width,
                    height: actual_height,
                    value,
                });
            }
        }
        
        segments
    }
    
    /// 计算分段的方差
    fn calculate_segments_variance(&self, segments: &[Segment], mean: f32) -> f32 {
        segments.iter()
            .map(|segment| {
                let diff = segment.value - mean;
                diff * diff * (segment.width * segment.height) as f32
            })
            .sum()
    }
    
    /// 计算分段相关性
    fn calculate_segmented_correlation(
         &self,
         image_integral: &[Vec<u64>],
         squared_image_integral: &[Vec<u64>],
         template_data: &SegmentedTemplateData,
         x: u32,
         y: u32,
         _template_width: u32,
         _template_height: u32,
     ) -> f32 {
        // 首先进行快速分段检查
        let fast_correlation = self.calculate_correlation_for_segments(
            image_integral,
            squared_image_integral,
            &template_data.fast_segments,
            template_data.fast_sum_squared_deviations,
            template_data.fast_mean,
            x,
            y,
        );
        
        // 如果快速检查不通过，直接返回
        if fast_correlation < template_data.expected_corr_fast {
            return fast_correlation;
        }
        
        // 进行慢速分段检查
        self.calculate_correlation_for_segments(
            image_integral,
            squared_image_integral,
            &template_data.slow_segments,
            template_data.slow_sum_squared_deviations,
            template_data.slow_mean,
            x,
            y,
        )
    }
    
    /// 为特定分段计算相关性
    fn calculate_correlation_for_segments(
        &self,
        image_integral: &[Vec<u64>],
        _squared_image_integral: &[Vec<u64>],
        segments: &[Segment],
        _template_sum_squared_deviations: f32,
        template_mean: f32,
        offset_x: u32,
        offset_y: u32,
    ) -> f32 {
        let mut template_values = Vec::new();
        let mut image_values = Vec::new();
        
        // 收集所有分段的值
        for segment in segments {
            let seg_x = offset_x + segment.x;
            let seg_y = offset_y + segment.y;
            
            let image_sum = self.sum_region(image_integral, seg_x, seg_y, segment.width, segment.height) as f32;
            let segment_pixels = (segment.width * segment.height) as f32;
            let segment_image_mean = image_sum / segment_pixels;
            
            template_values.push(segment.value);
            image_values.push(segment_image_mean);
        }
        
        // 计算图像值的均值
        let image_mean = image_values.iter().sum::<f32>() / image_values.len() as f32;
        
        // 计算Pearson相关系数
        let mut numerator = 0.0f32;
        let mut template_variance = 0.0f32;
        let mut image_variance = 0.0f32;
        
        for i in 0..template_values.len() {
            let template_dev = template_values[i] - template_mean;
            let image_dev = image_values[i] - image_mean;
            
            numerator += template_dev * image_dev;
            template_variance += template_dev * template_dev;
            image_variance += image_dev * image_dev;
        }
        
        let denominator = (template_variance * image_variance).sqrt();
        
        if denominator == 0.0 {
            0.0
        } else {
            let correlation = numerator / denominator;
            // 相关系数应该在[-1, 1]范围内
            correlation.max(-1.0).min(1.0)
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    #[test]
    fn test_integral_image() {
        let matcher = ImageMatcher::new();
        
        // 创建一个 3x3 的测试图像
        let mut image = ImageBuffer::new(3, 3);
        image.put_pixel(0, 0, Luma([1]));
        image.put_pixel(1, 0, Luma([2]));
        image.put_pixel(2, 0, Luma([3]));
        image.put_pixel(0, 1, Luma([4]));
        image.put_pixel(1, 1, Luma([5]));
        image.put_pixel(2, 1, Luma([6]));
        image.put_pixel(0, 2, Luma([7]));
        image.put_pixel(1, 2, Luma([8]));
        image.put_pixel(2, 2, Luma([9]));
        
        let (integral, _) = matcher.compute_integral_images(&image);
        
        // 测试整个图像的和
        assert_eq!(matcher.sum_region(&integral, 0, 0, 3, 3), 45);
        
        // 测试子区域的和
        assert_eq!(matcher.sum_region(&integral, 0, 0, 2, 2), 12); // 1+2+4+5
        assert_eq!(matcher.sum_region(&integral, 1, 1, 2, 2), 28); // 5+6+8+9
    }

    #[test]
    fn test_fft_template_matching() {
        // 创建简单的测试图像和模板
        let mut image = GrayImage::new(10, 10);
        let mut template = GrayImage::new(3, 3);
        
        // 填充测试数据
        for y in 0..10 {
            for x in 0..10 {
                image.put_pixel(x, y, Luma([((x + y) * 25) as u8]));
            }
        }
        
        for y in 0..3 {
            for x in 0..3 {
                template.put_pixel(x, y, Luma([((x + y) * 25) as u8]));
            }
        }
        
        // 执行匹配
        let mut matcher = ImageMatcher::new();
        matcher.prepare_template(&template, 10, 10, MatcherMode::FFT).unwrap();
        let results = matcher.match_by_fft(&image, 0.8).unwrap();
        
        // 应该找到至少一个匹配
        assert!(!results.is_empty());
        
        // 最佳匹配应该在(0,0)位置
        assert_eq!(results[0].x, 0);
        assert_eq!(results[0].y, 0);
        assert!(results[0].correlation > 0.9); // 高相关性
    }

    #[test]
    fn test_segmented_template_matching() {
        // 创建测试图像和模板
        let mut image = ImageBuffer::new(20, 20);
        let mut template = ImageBuffer::new(5, 5);
        
        // 创建一个简单的渐变模式
        for y in 0..20 {
            for x in 0..20 {
                image.put_pixel(x, y, Luma([((x + y) * 6) as u8]));
            }
        }
        
        // 模板是图像左上角的一部分
        for y in 0..5 {
            for x in 0..5 {
                template.put_pixel(x, y, Luma([((x + y) * 6) as u8]));
            }
        }
        
        // 执行分段匹配
        let mut matcher = ImageMatcher::new();
        matcher.prepare_template(&template, 5, 5, MatcherMode::Segmented).unwrap();
        let results = matcher.match_by_segmented(&image, 0.1).unwrap();
        
        // 应该找到至少一个匹配
        assert!(!results.is_empty());
        
        // 最佳匹配应该在(0,0)位置
        assert_eq!(results[0].x, 0);
        assert_eq!(results[0].y, 0);
        assert!(results[0].correlation > 0.3); // 分段匹配的相关性可能较低
    }

    #[test]
    fn test_segmented_template_matching_with_offset() {
        // 创建测试图像和模板
        let mut image = ImageBuffer::new(30, 30);
        let mut template = ImageBuffer::new(8, 8);
        
        // 创建一个简单的渐变模式
        for y in 0..30 {
            for x in 0..30 {
                let value = ((x + y) * 4) as u8;
                image.put_pixel(x, y, Luma([value]));
            }
        }
        
        // 模板是图像中间的一部分 (从位置 10,10 开始)
        for y in 0..8 {
            for x in 0..8 {
                let img_x = x + 10;
                let img_y = y + 10;
                let value = ((img_x + img_y) * 4) as u8;
                template.put_pixel(x, y, Luma([value]));
            }
        }
        
        // 执行分段匹配
        let mut matcher = ImageMatcher::new();
        matcher.prepare_template(&template, 8, 8, MatcherMode::Segmented).unwrap();
        let results = matcher.match_by_segmented(&image, 0.1).unwrap();
        
        assert!(!results.is_empty(), "应该找到至少一个匹配");
        
        // 检查是否在预期位置找到了匹配
        let found_at_expected = results.iter().any(|r| r.x == 10 && r.y == 10 && r.correlation > 0.1);
        assert!(found_at_expected, "应该在位置 (10,10) 找到匹配，相关性 > 0.1");
    }

    #[test]
    fn test_segmented_template_matching_no_match() {
        // 创建完全不同的图像和模板
        let mut image = ImageBuffer::new(15, 15);
        let mut template = ImageBuffer::new(5, 5);
        
        // 图像是纯白色
        for y in 0..15 {
            for x in 0..15 {
                image.put_pixel(x, y, Luma([255]));
            }
        }
        
        // 模板是纯黑色
        for y in 0..5 {
            for x in 0..5 {
                template.put_pixel(x, y, Luma([0]));
            }
        }
        
        // 执行分段匹配
        let mut matcher = ImageMatcher::new();
        matcher.prepare_template(&template, 5, 5, MatcherMode::Segmented).unwrap();
        let results = matcher.match_by_segmented(&image, 0.8).unwrap();
        
        // 由于阈值较高且图像完全不匹配，应该没有结果
        assert!(results.is_empty() || results[0].correlation < 0.8);
    }

    #[test]
    fn test_segmented_prepare_template() {
        // 测试分段模板准备功能
        let mut template = ImageBuffer::new(8, 8);
        
        // 创建一个有变化的模板
        for y in 0..8 {
            for x in 0..8 {
                let value = ((x * y) % 256) as u8;
                template.put_pixel(x, y, Luma([value]));
            }
        }
        
        let matcher = ImageMatcher::new();
        let template_data = matcher.prepare_segmented_template(&template).unwrap();
        
        // 验证分段数据
        assert!(!template_data.fast_segments.is_empty());
        assert!(!template_data.slow_segments.is_empty());
        assert_eq!(template_data.template_width, 8);
        assert_eq!(template_data.template_height, 8);
        assert!(template_data.fast_sum_squared_deviations >= 0.0);
        assert!(template_data.slow_sum_squared_deviations >= 0.0);
        
        // 慢速分段应该比快速分段更多
        assert!(template_data.slow_segments.len() >= template_data.fast_segments.len());
    }
}

/// 分段模板数据
#[derive(Debug, Clone)]
struct SegmentedTemplateData {
    /// 快速分段（粗糙分割，段数少）
    fast_segments: Vec<Segment>,
    /// 慢速分段（精细分割，段数多）
    slow_segments: Vec<Segment>,
    /// 模板尺寸
    template_width: u32,
    template_height: u32,
    /// 快速分段的平方差之和
    fast_sum_squared_deviations: f32,
    /// 慢速分段的平方差之和
    slow_sum_squared_deviations: f32,
    /// 快速分段的均值
    fast_mean: f32,
    /// 慢速分段的均值
    slow_mean: f32,
    /// 期望相关性阈值
    expected_corr_fast: f32,
}

/// 图像段
#[derive(Debug, Clone)]
struct Segment {
    /// 段的起始 x 坐标
    x: u32,
    /// 段的起始 y 坐标
    y: u32,
    /// 段的宽度
    width: u32,
    /// 段的高度
    height: u32,
    /// 段的平均值
    value: f32,
}
