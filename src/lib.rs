//! Rust 图像匹配库
//!
//! 提供基于 FFT 的归一化互相关（NCC）模板匹配实现。
//! 核心算法来源于 J.P. Lewis 的论文：
//! "Fast Normalized Cross-Correlation" (http://scribblethink.org/Work/nvisionInterface/vi95_lewis.pdf)
//!
//! ## 使用方法
//!
//! ```rust,no_run
//! use image_matching_rs::{ImageMatcher, MatcherMode};
//! use image::{ImageBuffer, Luma, DynamicImage};
//!
//! // 创建模板和图像
//! let template = DynamicImage::ImageLuma8(ImageBuffer::<Luma<u8>, Vec<u8>>::new(10, 10));
//! let image = DynamicImage::ImageLuma8(ImageBuffer::<Luma<u8>, Vec<u8>>::new(100, 100));
//!
//! let matcher = ImageMatcher::new_from_image(template, MatcherMode::FFT { src_width: 100, src_height: 100 }, None);
//! let results = matcher.matching(image, 0.8, None).unwrap();
//! ```
//!
mod matcher;
pub use matcher::ImageMatcher;

mod mode;
pub use mode::MatcherMode;

mod filter;
pub use filter::MatchHitFilter;

mod result;
pub use result::MatchResults;
pub use result::MatchHit;

mod error;
pub use error::ImageError;

mod template;
