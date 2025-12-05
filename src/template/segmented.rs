use super::generate_search_coordinates;
use crate::error::ImageError;
use crate::result::MatchHit;
use crate::template::TemplateData;
use crate::template::integral::{IntegralImage, IntegralImages};
use image::{GrayImage, ImageBuffer, Luma};
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

#[derive(Debug, Clone)]
/// 分段模板匹配数据
///
/// 使用分段统计特征进行快速匹配筛选。
pub struct SegmentedTemplateData {
    /// 模板宽度
    pub template_width: u32,
    /// 模板高度
    pub template_height: u32,
    fast_config: Config,
    slow_config: Config,
}

#[derive(Debug, Clone, Copy)]
/// 分段类型
pub enum SegmentType {
    /// 快速分段（较粗略）
    Fast,
    /// 慢速分段（较精细）
    Slow,
}

struct ImageStatistics {
    pub mean: f64,
    pub avg_deviation: f64,
}

#[derive(Debug, Clone)]
struct Config {
    template_segments: Vec<(u32, u32, u32, u32, f64)>,
    sum_squared_deviations: f64,
    expected_corr: f64,
    segments_mean: f64,
}

struct AdjustedThresholds {
    pub fast_threshold: f64,
    pub slow_threshold: f64,
}

impl SegmentedTemplateData {
    fn calculate_template_statistics(template: &ImageBuffer<Luma<u8>, Vec<u8>>) -> ImageStatistics {
        let (width, height) = template.dimensions();
        let total_pixels = (width * height) as f64;
        let sum: u64 = (0..height)
            .flat_map(|y| (0..width).map(move |x| template.get_pixel(x, y)[0] as u64))
            .sum();
        let mean = sum as f64 / total_pixels;
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

    fn recursive_binary_segmentation(
        template: &ImageBuffer<Luma<u8>, Vec<u8>>,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        min_std_dev: f64,
        segments: &mut Vec<(u32, u32, u32, u32, f64)>,
    ) {
        let (mean, std_dev) = Self::calculate_region_statistics(template, x, y, width, height);
        if std_dev < min_std_dev || width < 4 || height < 4 {
            segments.push((x, y, width, height, mean));
            return;
        }
        if width >= height {
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

    fn should_merge_segments(
        seg1: &(u32, u32, u32, u32, f64),
        seg2: &(u32, u32, u32, u32, f64),
    ) -> bool {
        let (x1, y1, w1, h1, mean1) = *seg1;
        let (x2, y2, w2, h2, mean2) = *seg2;
        let adjacent = (x1 + w1 == x2 && y1 == y2 && h1 == h2)
            || (x1 == x2 && y1 + h1 == y2 && w1 == w2)
            || (x2 + w2 == x1 && y1 == y2 && h1 == h2)
            || (x1 == x2 && y2 + h2 == y1 && w1 == w2);
        let mean_similar = (mean1 - mean2).abs() < (mean1 + mean2) * 0.05;
        adjacent && mean_similar
    }

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
        let area1 = (w1 * h1) as f64;
        let area2 = (w2 * h2) as f64;
        let total_area = area1 + area2;
        let new_mean = (mean1 * area1 + mean2 * area2) / total_area;
        (min_x, min_y, new_width, new_height, new_mean)
    }

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
        let variance = sum_squared_deviations / total_area as f64;
        let expected_corr = (variance / (variance + 1.0)).sqrt();
        (sum_squared_deviations, expected_corr)
    }

    fn create_template_segments(
        template: &ImageBuffer<Luma<u8>, Vec<u8>>,
        mean_template_value: f64,
        avg_deviation_of_template: f64,
        segment_type: SegmentType,
    ) -> Config {
        let (template_width, template_height) = template.dimensions();
        let mut picture_segments: Vec<(u32, u32, u32, u32, f64)> = Vec::new();
        let (max_segments, min_std_dev_multiplier) = match segment_type {
            SegmentType::Fast => (25, 0.8),
            SegmentType::Slow => (100, 0.6),
        };
        let mut segments_mean = 0.0;
        let mut min_std_dev = avg_deviation_of_template * min_std_dev_multiplier;
        while picture_segments.len() > max_segments || picture_segments.is_empty() {
            picture_segments.clear();
            segments_mean = 0.0;
            Self::recursive_binary_segmentation(
                template,
                0,
                0,
                template_width,
                template_height,
                min_std_dev,
                &mut picture_segments,
            );
            if !picture_segments.is_empty() {
                let sum_means: f64 = picture_segments.iter().map(|(_, _, _, _, mean)| mean).sum();
                segments_mean = sum_means / picture_segments.len() as f64;
            }
            if picture_segments.len() > max_segments {
                min_std_dev *= 1.1;
            } else if picture_segments.is_empty() {
                min_std_dev *= 0.9;
            }
            if min_std_dev > avg_deviation_of_template * 2.0 || min_std_dev < 0.1 {
                break;
            }
        }
        if picture_segments.is_empty() {
            picture_segments.push((0, 0, template_width, template_height, mean_template_value));
            segments_mean = mean_template_value;
        }
        let merged_segments = Self::merge_similar_segments(picture_segments);
        let (segment_sum_squared_deviations, expected_corr) =
            Self::calculate_segment_statistics(&merged_segments, segments_mean);

        Config {
            template_segments: merged_segments,
            sum_squared_deviations: segment_sum_squared_deviations,
            expected_corr,
            segments_mean,
        }
    }

    pub fn new(template: GrayImage) -> Self {
        let template_stats = Self::calculate_template_statistics(&template);

        let fast_config = Self::create_template_segments(
            &template,
            template_stats.mean,
            template_stats.avg_deviation,
            SegmentType::Fast,
        );

        let slow_config = Self::create_template_segments(
            &template,
            template_stats.mean,
            template_stats.avg_deviation,
            SegmentType::Slow,
        );

        SegmentedTemplateData {
            template_width: template.width(),
            template_height: template.height(),
            fast_config,
            slow_config,
        }
    }
}

impl SegmentedTemplateData {
    fn calculate_adjusted_thresholds(&self, threshold: f64) -> AdjustedThresholds {
        let base = threshold.min(0.95);

        AdjustedThresholds {
            fast_threshold: base * self.fast_config.expected_corr - 0.0001,
            slow_threshold: base * self.slow_config.expected_corr - 0.0001,
        }
    }

    fn calculate_segment_correlation(
        &self,
        image_integral: &IntegralImage,
        config: &Config,
        x: u32,
        y: u32,
    ) -> f64 {
        let mut numerator = 0.0f64;

        let (template_width, template_height) = self.template_dimensions();

        let total_image_sum = image_integral.sum_region(
            x as usize,
            y as usize,
            template_width as usize,
            template_height as usize,
        );
        let template_size = (template_width * template_height) as f64;
        let image_mean = total_image_sum / template_size;

        let mut image_segment_sum_squared_deviations = 0.0f64;
        for &(seg_x, seg_y, seg_width, seg_height, segment_mean) in &config.template_segments {
            let region_sum = image_integral.sum_region(
                x as usize + seg_x as usize,
                y as usize + seg_y as usize,
                seg_width as usize,
                seg_height as usize,
            );
            let region_size = (seg_width * seg_height) as f64;
            let region_mean = region_sum / region_size;
            numerator +=
                (segment_mean - config.segments_mean) * (region_mean - image_mean) * region_size;
            image_segment_sum_squared_deviations +=
                (region_mean - image_mean) * (region_mean - image_mean) * region_size;
        }

        let denominator =
            (config.sum_squared_deviations * image_segment_sum_squared_deviations).sqrt();
        if denominator == 0.0 {
            0.0
        } else {
            (numerator / denominator).clamp(-1.0, 1.0)
        }
    }

    fn calculate_single_segmented_correlation(
        &self,
        image_integral: &IntegralImage,
        x: u32,
        y: u32,
        fast_threshold: f64,
    ) -> f64 {
        let fast_corr = self.calculate_segment_correlation(image_integral, &self.fast_config, x, y);

        if fast_corr < fast_threshold {
            return fast_corr;
        }

        self.calculate_segment_correlation(image_integral, &self.slow_config, x, y)
    }

    fn calculate_segmented_correlations(
        &self,
        integral_images: &IntegralImages,
        thresholds: &AdjustedThresholds,
        image_width: u32,
        image_height: u32,
    ) -> Result<Vec<MatchHit>, ImageError> {
        let coords = generate_search_coordinates(
            image_width,
            image_height,
            self.template_width,
            self.template_height,
        );

        let list: Vec<MatchHit> = coords
            .par_iter()
            .with_min_len(1024)
            .map(|&(x, y)| {
                let correlation = self.calculate_single_segmented_correlation(
                    &integral_images.integral,
                    x,
                    y,
                    thresholds.fast_threshold,
                );
                (x, y, correlation)
            })
            .filter(|&(_, _, correlation)| correlation >= thresholds.slow_threshold)
            .map(|(x, y, correlation)| MatchHit { x, y, correlation })
            .collect();

        Ok(list)
    }
}

impl TemplateData for SegmentedTemplateData {
    fn template_dimensions(&self) -> (u32, u32) {
        (self.template_width, self.template_height)
    }

    fn perform_matching(
        &self,
        img: GrayImage,
        threshold: f64,
    ) -> Result<Vec<MatchHit>, ImageError> {
        let integral_img = IntegralImages::new(&img, true);

        let adjusted_thresholds = self.calculate_adjusted_thresholds(threshold);

        self.calculate_segmented_correlations(
            &integral_img,
            &adjusted_thresholds,
            img.width(),
            img.height(),
        )
    }
}
