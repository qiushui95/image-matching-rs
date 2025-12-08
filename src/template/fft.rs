use super::generate_search_coordinates;
use crate::error::ImageError;
use crate::result::MatchHit;
use crate::template::integral::IntegralImages;
use crate::template::TemplateData;
use image::{GrayImage, ImageBuffer, Luma};
use num_complex::Complex;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::IntoParallelRefIterator;
use rustfft::FftPlanner;
use std::cmp::max;

#[derive(Debug, Clone)]
/// FFT 模板匹配数据
///
/// 存储用于 FFT 归一化互相关匹配的预计算数据。
pub struct FFTTemplateData {
    /// 模板的频域共轭表示
    pub template_conj_freq: Vec<Complex<f64>>,
    /// 模板的平方差之和（预计算用于分母）
    pub template_sum_squared_deviations: f64,
    /// 预期的源图像宽度
    pub src_width: u32,
    /// 预期的源图像高度
    pub src_height: u32,
    /// 模板宽度
    pub template_width: u32,
    /// 模板高度
    pub template_height: u32,
    /// FFT 计算所需的填充大小（2 的幂次）
    pub padded_size: usize,
}

impl FFTTemplateData {
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

    fn create_zero_mean_template(template_vec: &[Vec<u8>], mean: f64) -> Vec<Vec<f64>> {
        template_vec
            .iter()
            .map(|row| row.iter().map(|&pixel| pixel as f64 - mean).collect())
            .collect()
    }

    fn calculate_sum_squared_deviations(zero_mean_template: &[Vec<f64>]) -> f64 {
        zero_mean_template
            .iter()
            .flat_map(|row| row.iter())
            .map(|&val| val * val)
            .sum()
    }

    fn create_fft_template_conjugate(
        zero_mean_template: &[Vec<f64>],
        template_width: u32,
        template_height: u32,
        padded_size: usize,
    ) -> Vec<Complex<f64>> {
        let padded_len = padded_size * padded_size;

        let mut template_padded = vec![Complex::new(0.0, 0.0); padded_len];
        for y in 0..template_height as usize {
            for x in 0..template_width as usize {
                let pixel_value = zero_mean_template[y][x];
                template_padded[y * padded_size + x] = Complex::new(pixel_value, 0.0);
            }
        }
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(padded_len);
        fft.process(&mut template_padded);
        template_padded.iter().map(|&val| val.conj()).collect()
    }

    /// 构建 FFT 模板数据，预计算频域模板与方差项
    pub fn new(template: GrayImage, src_width: u32, src_height: u32) -> Self {
        let (template_width, template_height) = template.dimensions();
        let required_width = src_width + template_width - 1;
        let required_height = src_height + template_height - 1;
        let padded_width = required_width.next_power_of_two();
        let padded_height = required_height.next_power_of_two();
        let padded_size = max(padded_width, padded_height) as usize;

        let template_vec = Self::image_buffer_to_2d_vec(&template);
        let template_mean = Self::calculate_image_mean(&template_vec);
        let zero_mean_template = Self::create_zero_mean_template(&template_vec, template_mean);
        let template_sum_squared_deviations =
            Self::calculate_sum_squared_deviations(&zero_mean_template);
        let template_conj_freq = Self::create_fft_template_conjugate(
            &zero_mean_template,
            template_width,
            template_height,
            padded_size,
        );

        FFTTemplateData {
            template_conj_freq,
            template_sum_squared_deviations,
            src_width,
            src_height,
            template_width,
            template_height,
            padded_size,
        }
    }
}

impl FFTTemplateData {
    /// 将输入图像转换为零均值二维数组
    fn create_zero_mean_image(&self, image: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Vec<Vec<f64>> {
        let (width, height) = image.dimensions();
        let sum: u64 = (0..height)
            .flat_map(|y| (0..width).map(move |x| image.get_pixel(x, y)[0] as u64))
            .sum();

        let mean = sum as f64 / (width * height) as f64;

        (0..height)
            .map(|y| {
                (0..width)
                    .map(|x| image.get_pixel(x, y)[0] as f64 - mean)
                    .collect()
            })
            .collect()
    }

    /// 在频域执行卷积并返回反变换后的相关结果矩阵
    fn perform_convolution(&self, img: &GrayImage) -> Vec<Complex<f64>> {
        let padded_size = self.padded_size;
        let padded_len = padded_size * padded_size;

        let mut image_padded = vec![Complex::new(0.0, 0.0); padded_len];

        let zero_mean_image = self.create_zero_mean_image(img);

        for y in 0..self.src_height as usize {
            for x in 0..self.src_width as usize {
                let pixel_value = zero_mean_image[y][x];
                image_padded[y * padded_size + x] = Complex::new(pixel_value, 0.0);
            }
        }

        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(padded_len);
        fft.process(&mut image_padded);
        let product_freq: Vec<Complex<f64>> = image_padded
            .par_iter()
            .zip(self.template_conj_freq.par_iter())
            .map(|(&img_val, &tmpl_val)| img_val * tmpl_val)
            .collect();
        let mut fft_result = product_freq;
        let ifft = planner.plan_fft_inverse(padded_len);
        ifft.process(&mut fft_result);
        fft_result
    }

    /// 计算所有滑窗位置的相关系数并按阈值筛选
    fn calculate_correlations(
        &self,
        integral_images: &IntegralImages,
        result: &[Complex<f64>],
        threshold: f64,
    ) -> Result<Vec<MatchHit>, ImageError> {
        let coords = generate_search_coordinates(
            self.src_width,
            self.src_height,
            self.template_width,
            self.template_height,
        );

        let list: Vec<MatchHit> = coords
            .par_iter()
            .with_min_len(1024)
            .map(|&(x, y)| {
                let correlation = self.calculate_single_correlation(
                    integral_images,
                    x as usize,
                    y as usize,
                    result,
                );
                (x, y, correlation)
            })
            .filter(|&(_, _, correlation)| correlation >= threshold)
            .map(|(x, y, correlation)| MatchHit {
                x,
                y,
                width: self.template_width,
                height: self.template_height,
                correlation,
            })
            .collect();

        Ok(list)
    }

    /// 计算单个滑窗位置的相关系数
    fn calculate_single_correlation(
        &self,
        integral_images: &IntegralImages,
        x: usize,
        y: usize,
        result: &[Complex<f64>],
    ) -> f64 {
        let padded_size = self.padded_size;
        let padded_len = padded_size * padded_size;
        let numerator = result[y * padded_size + x].re / padded_len as f64;

        let sum_image_region = integral_images.integral.sum_region(
            x,
            y,
            self.template_width as usize,
            self.template_height as usize,
        );

        let sum_squared_image_region = integral_images.squared_integral.sum_region(
            x,
            y,
            self.template_width as usize,
            self.template_height as usize,
        );

        let template_size = (self.template_width * self.template_height) as f64;
        let image_mean_squared = (sum_image_region * sum_image_region) / template_size;
        let image_sum_squared_deviations = sum_squared_image_region - image_mean_squared;

        let template_sum_squared_deviations = self.template_sum_squared_deviations;
        let denominator = (template_sum_squared_deviations * image_sum_squared_deviations).sqrt();

        if denominator == 0.0 {
            return 0.0;
        }

        let correlation = numerator / denominator;
        if !(-2.0..=2.0).contains(&correlation) {
            0.0
        } else {
            correlation
        }
    }
}

impl TemplateData for FFTTemplateData {
    fn template_dimensions(&self) -> (u32, u32) {
        (self.template_width, self.template_height)
    }

    fn check_img(&self, img: &GrayImage) -> Result<(), ImageError> {
        TemplateData::common_check_img(self, img)?;

        let (img_width, img_height) = img.dimensions();

        if img_width != self.src_width || img_height != self.src_height {
            return Err(ImageError::DimensionsMismatch(
                img_width,
                img_height,
                self.src_width,
                self.src_height,
            ));
        }

        Ok(())
    }

    fn perform_matching(
        &self,
        img: GrayImage,
        threshold: f64,
    ) -> Result<Vec<MatchHit>, ImageError> {
        let integral_img = IntegralImages::new(&img, false);
        let convolution = self.perform_convolution(&img);

        self.calculate_correlations(&integral_img, &convolution, threshold)
    }
}
