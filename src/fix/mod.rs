pub mod matcher_mode;
pub mod matcher_single_result;
pub mod matcher_result_filter;
pub mod matcher_result;
pub mod template_data;
pub mod fft_template_data;
pub mod segmented_template_data;
pub mod adjusted_thresholds;
pub mod image_statistics;
pub mod integral_images;
pub mod image_matcher;

pub use matcher_mode::MatcherMode;
pub use matcher_single_result::MatcherSingleResult;
pub use image_matcher::ImageMatcher;
pub use matcher_result::MatcherResult;
pub use matcher_result_filter::MatcherResultFilter;
