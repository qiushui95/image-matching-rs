use image::{GenericImageView, ImageBuffer, Luma};
use num_complex::Complex;
use rustfft::FftPlanner;
use std::cmp::max;

#[derive(Debug, Clone)]
pub struct FFTTemplateData {
    pub template_conj_freq: Vec<Complex<f64>>,
    pub template_sum_squared_deviations: f64,
    pub template_width: u32,
    pub template_height: u32,
    pub padded_size: u32,
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
        padded_size: u32,
    ) -> Vec<Complex<f64>> {
        let mut template_padded =
            vec![Complex::new(0.0, 0.0); (padded_size * padded_size) as usize];
        for y in 0..template_height {
            for x in 0..template_width {
                let pixel_value = zero_mean_template[y as usize][x as usize];
                template_padded[y as usize * padded_size as usize + x as usize] =
                    Complex::new(pixel_value, 0.0);
            }
        }
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward((padded_size * padded_size) as usize);
        fft.process(&mut template_padded);
        template_padded.iter().map(|&val| val.conj()).collect()
    }

    pub fn new(template: &ImageBuffer<Luma<u8>, Vec<u8>>, src_width: u32, src_height: u32) -> Self {
        let (template_width, template_height) = template.dimensions();
        let required_width = src_width + template_width - 1;
        let required_height = src_height + template_height - 1;
        let padded_width = required_width.next_power_of_two();
        let padded_height = required_height.next_power_of_two();
        let padded_size = max(padded_width, padded_height);

        let template_vec = Self::image_buffer_to_2d_vec(template);
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
            template_width,
            template_height,
            padded_size,
        }
    }
}
