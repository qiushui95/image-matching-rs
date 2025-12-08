use crate::MatcherMode;
use crate::error::ImageError;
use crate::filter::MatchHitFilter;
use crate::result::MatchResults;
use crate::template::{TemplateData, TemplateDataEnum};
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView};
use rayon::slice::ParallelSliceMut;

/// 图像匹配器
///
/// 提供基于模板匹配的图像识别功能。
pub struct ImageMatcher {
    template_data: TemplateDataEnum,
    /// 模板宽度
    pub template_width: u32,
    /// 模板高度
    pub template_height: u32,
}

impl ImageMatcher {
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

    fn set_global_rayon_threads(num_threads: usize) {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global();
    }

    /// 从图像创建匹配器
    ///
    /// # 参数
    ///
    /// * `img` - 模板图像
    /// * `mode` - 匹配模式（FFT 或 Segmented）
    /// * `resize_width` - 可选的缩放宽度，如果提供，模板将被缩放到指定宽度
    pub fn new_from_image(img: DynamicImage, mode: MatcherMode, resize_width: Option<u32>) -> Self {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        Self::set_global_rayon_threads(threads);

        let img = Self::resize_image_if_needed(img, resize_width);
        let template_image = img.to_luma8();
        let (template_width, template_height) = template_image.dimensions();

        let template_data = TemplateDataEnum::new(template_image, mode);

        Self {
            template_data,
            template_width,
            template_height,
        }
    }

    /// 执行图像匹配
    ///
    /// # 参数
    ///
    /// * `img` - 待匹配的源图像
    /// * `threshold` - 匹配阈值，范围 [0.0, 1.0]
    /// * `filter` - 可选的结果过滤器，用于去除重叠结果
    ///
    /// # 返回值
    ///
    /// 返回匹配结果，包含最佳匹配和所有匹配列表。
    pub fn matching(
        &self,
        img: DynamicImage,
        threshold: f64,
        filter: Option<MatchHitFilter>,
    ) -> Result<MatchResults, ImageError> {
        let image = img.to_luma8();

        self.template_data.check_img(&image)?;

        let template_width = self.template_width;
        let template_height = self.template_height;

        let (image_width, image_height) = image.dimensions();
        if template_width > image_width {
            return Err(ImageError::TemplateWidthTooLarge);
        }

        if template_height > image_height {
            return Err(ImageError::TemplateHeightTooLarge);
        }

        let mut list = self.template_data.perform_matching(image, threshold)?;

        list.par_sort_unstable_by(|a, b| b.cmp(a));

        let Some(filter) = filter else {
            return MatchResults::new(list);
        };

        let results = filter.filter_results(list);

        MatchResults::new(results)
    }
}
