use crate::template_data::TemplateData;
use crate::MatcherMode;
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView};

pub struct ImageMatcher {
    template_data: TemplateData,
    mode: MatcherMode,
    pub template_width: u32,
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

    pub fn new_from_image(img: DynamicImage, mode: MatcherMode, resize_width: Option<u32>) -> Self {
        let img = Self::resize_image_if_needed(img, resize_width);
        let template_image = img.to_luma8();
        let (template_width, template_height) = template_image.dimensions();

        let template_data = TemplateData::new(&template_image, mode);

        Self {
            template_data,
            mode,
            template_width,
            template_height,
        }
    }

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
}
