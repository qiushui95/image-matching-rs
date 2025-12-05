use image::{DynamicImage, GenericImageView, ImageBuffer, Luma, imageops::FilterType};
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};
use std::cmp::max;
use std::error::Error;

use super::matcher_mode::MatcherMode;
use super::matcher_result::MatcherResult;
use super::matcher_result_filter::MatcherResultFilter;
use super::matcher_single_result::MatcherSingleResult;
use super::template_data::TemplateData;
use super::fft_template_data::FFTTemplateData;
use super::segmented_template_data::{SegmentType, SegmentedTemplateData};
use super::adjusted_thresholds::AdjustedThresholds;
use super::image_statistics::ImageStatistics;
use super::integral_images::IntegralImages;

pub struct ImageMatcher {
    template_data: TemplateData,
    mode: MatcherMode,
    pub template_width: u32,
    pub template_height: u32,
}

impl ImageMatcher {

    

    pub fn matching(
        &self,
        img: DynamicImage,
        threshold: f64,
        filter: Option<MatcherResultFilter>,
    ) -> Result<MatcherResult, Box<dyn Error>> {
        let image = img.to_luma8();

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

        let mut list = match &self.template_data {
            TemplateData::FFT { data } => Self::perform_fft_matching(data, image, threshold),
            TemplateData::Segmented { data } => Self::perform_segmented_matching(data, image, threshold),
        }?;

        list.sort_by(|a, b| a.cmp(b).reverse());

        let Some(filter) = filter else { return MatcherResult::new(&self.template_data, list); };

        let mut results = vec![];
        for item in list {
            let mut need_filter = false;
            for exist in results.iter() {
                if filter.need_filter(&item, exist) { need_filter = true; break; }
            }
            if need_filter { continue; }
            results.push(item);
        }

        MatcherResult::new(&self.template_data, results)
    }

    fn perform_fft_matching(
        template_data: &FFTTemplateData,
        image: ImageBuffer<Luma<u8>, Vec<u8>>,
        threshold: f64,
    ) -> Result<Vec<MatcherSingleResult>, Box<dyn Error>> {
        let (image_width, image_height) = image.dimensions();
        if template_data.template_width > image_width { return Err("模板宽度大于图像宽度".into()); }
        if template_data.template_height > image_height { return Err("模板高度大于图像高度".into()); }
        let integral_images = Self::compute_integral_images(&image);
        let zero_mean_image = Self::create_zero_mean_image(&image);
        let fft_result = Self::perform_fft_convolution(
            &zero_mean_image,
            image_width,
            image_height,
            template_data,
        );
        Self::calculate_fft_correlations(
            &integral_images,
            template_data,
            &fft_result,
            image_width,
            image_height,
            threshold,
        )
    }

    fn perform_segmented_matching(
        template_data: &SegmentedTemplateData,
        image: ImageBuffer<Luma<u8>, Vec<u8>>,
        threshold: f64,
    ) -> Result<Vec<MatcherSingleResult>, Box<dyn Error>> {
        let (template_width, template_height) = (template_data.template_width, template_data.template_height);
        let (image_width, image_height) = image.dimensions();
        if template_width > image_width { return Err("模板宽度大于图像宽度".into()); }
        if template_height > image_height { return Err("模板高度大于图像高度".into()); }
        let integral_images = Self::compute_integral_images(&image);
        let adjusted_thresholds = Self::calculate_adjusted_thresholds(template_data, threshold);
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







    fn create_zero_mean_image(image: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Vec<Vec<f64>> {
        let (width, height) = image.dimensions();
        let sum: u64 = (0..height).flat_map(|y| (0..width).map(move |x| image.get_pixel(x, y)[0] as u64)).sum();
        let mean = sum as f64 / (width * height) as f64;
        (0..height)
            .map(|y| (0..width).map(|x| image.get_pixel(x, y)[0] as f64 - mean).collect())
            .collect()
    }

    fn perform_fft_convolution(
        zero_mean_image: &[Vec<f64>],
        image_width: u32,
        image_height: u32,
        template_data: &FFTTemplateData,
    ) -> Vec<Complex<f64>> {
        let mut image_padded = vec![Complex::new(0.0, 0.0); (template_data.padded_size * template_data.padded_size) as usize];
        for y in 0..image_height {
            for x in 0..image_width {
                let pixel_value = zero_mean_image[y as usize][x as usize];
                image_padded[y as usize * template_data.padded_size as usize + x as usize] = Complex::new(pixel_value, 0.0);
            }
        }
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward((template_data.padded_size * template_data.padded_size) as usize);
        fft.process(&mut image_padded);
        let product_freq: Vec<Complex<f64>> = image_padded
            .iter()
            .zip(template_data.template_conj_freq.iter())
            .map(|(&img_val, &tmpl_val)| img_val * tmpl_val)
            .collect();
        let mut fft_result = product_freq;
        let ifft = planner.plan_fft_inverse((template_data.padded_size * template_data.padded_size) as usize);
        ifft.process(&mut fft_result);
        fft_result
    }

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
        IntegralImages { integral, squared_integral }
    }

    fn sum_region(integral: &[Vec<u64>], x: u32, y: u32, width: u32, height: u32) -> u64 {
        let x1 = x as usize;
        let y1 = y as usize;
        let x2 = (x + width) as usize;
        let y2 = (y + height) as usize;
        integral[y2][x2] + integral[y1][x1] - integral[y1][x2] - integral[y2][x1]
    }

    fn calculate_fft_correlations(
        integral_images: &IntegralImages,
        template_data: &FFTTemplateData,
        fft_result: &[Complex<f64>],
        image_width: u32,
        image_height: u32,
        threshold: f64,
    ) -> Result<Vec<MatcherSingleResult>, Box<dyn Error>> {
        let coords: Vec<(u32, u32)> = (0..=(image_height - template_data.template_height))
            .flat_map(|y| (0..=(image_width - template_data.template_width)).map(move |x| (x, y)))
            .collect();

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

    fn calculate_single_fft_correlation(
        image_integral: &[Vec<u64>],
        squared_image_integral: &[Vec<u64>],
        template_data: &FFTTemplateData,
        x: u32,
        y: u32,
        fft_result: &[Complex<f64>],
    ) -> f64 {
        let numerator = fft_result[y as usize * template_data.padded_size as usize + x as usize].re
            / (template_data.padded_size * template_data.padded_size) as f64;

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

        let denominator = (template_data.template_sum_squared_deviations * image_sum_squared_deviations).sqrt();

        if denominator == 0.0 { 0.0 } else {
            let correlation = numerator / denominator;
            if correlation > 2.0 || correlation < -2.0 { 0.0 } else { correlation }
        }
    }

    fn calculate_adjusted_thresholds(
        template_data: &SegmentedTemplateData,
        threshold: f64,
    ) -> AdjustedThresholds {
        let base = threshold.min(0.95);
        AdjustedThresholds {
            fast_threshold: base * template_data.expected_corr_fast - 0.0001,
            slow_threshold: base * template_data.expected_corr_slow - 0.0001,
        }
    }

    fn calculate_segmented_correlations(
        integral_images: &IntegralImages,
        template_data: &SegmentedTemplateData,
        thresholds: &AdjustedThresholds,
        image_width: u32,
        image_height: u32,
        template_width: u32,
        template_height: u32,
    ) -> Result<Vec<MatcherSingleResult>, Box<dyn Error>> {
        let coords: Vec<(u32, u32)> = (0..=(image_height - template_height))
            .flat_map(|y| (0..=(image_width - template_width)).map(move |x| (x, y)))
            .collect();

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

    fn calculate_single_segmented_correlation(
        image_integral: &[Vec<u64>],
        squared_image_integral: &[Vec<u64>],
        template_data: &SegmentedTemplateData,
        x: u32,
        y: u32,
        fast_threshold: f64,
    ) -> f64 {
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
        if fast_corr < fast_threshold { return fast_corr; }
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

    fn calculate_segment_correlation(
        image_integral: &[Vec<u64>],
        _squared_image_integral: &[Vec<u64>],
        segments: &[(u32, u32, u32, u32, f64)],
        template_width: u32,
        template_height: u32,
        segment_sum_squared_deviations: f64,
        segments_mean: f64,
        x: u32,
        y: u32,
    ) -> f64 {
        let mut numerator = 0.0f64;
        let total_image_sum = Self::sum_region(image_integral, x, y, template_width, template_height) as f64;
        let template_size = (template_width * template_height) as f64;
        let image_mean = total_image_sum / template_size;

        let mut image_segment_sum_squared_deviations = 0.0f64;
        for &(seg_x, seg_y, seg_width, seg_height, segment_mean) in segments {
            let region_sum = Self::sum_region(image_integral, x + seg_x, y + seg_y, seg_width, seg_height) as f64;
            let region_size = (seg_width * seg_height) as f64;
            let region_mean = region_sum / region_size;
            numerator += (segment_mean - segments_mean) * (region_mean - image_mean) * region_size;
            image_segment_sum_squared_deviations += (region_mean - image_mean) * (region_mean - image_mean) * region_size;
        }

        let denominator = (segment_sum_squared_deviations * image_segment_sum_squared_deviations).sqrt();
        if denominator == 0.0 { 0.0 } else { (numerator / denominator).clamp(-1.0, 1.0) }
    }














}
