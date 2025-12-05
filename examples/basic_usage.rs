use image::{DynamicImage, GenericImageView};
use image_matching_rs::{ImageMatcher, MatchHitFilter, MatcherMode};
use std::error::Error;
use std::time;

fn main() -> Result<(), Box<dyn Error>> {
    let template_list = vec![
        "template50x50.png",
        "template100x100.png",
        "template150x150.png",
        "template200x200.png",
    ];

    let screen_list = vec![
        "screen1024x768.png",
        "screen1920x1080.png",
        "screen2560x1440.png",
    ];

    for &template_path in &template_list {
        for &screen_path in &screen_list {
            start_all_match(template_path, screen_path)?;
        }
    }

    Ok(())
}

#[derive(Debug, serde::Deserialize, Copy, Clone)]
struct MatchResult {
    x: u32,
    y: u32,
    width: u32,
}
fn get_best_result(
    screen_path: String,
    template_width: u32,
) -> Result<MatchResult, Box<dyn Error>> {
    let json_path = screen_path.replace(".png", ".json");

    let json_content = std::fs::read_to_string(json_path)?;
    let list: Vec<MatchResult> = serde_json::from_str(&json_content)?;

    let Some(best_result) = list
        .iter()
        .filter(|item| item.width == template_width)
        .next()
    else {
        return Err("未找到匹配的结果".into());
    };

    Ok(*best_result)
}

fn start_all_match(template_path: &str, screen_path: &str) -> Result<(), Box<dyn Error>> {
    println!("{}", "+".repeat(120));

    // 加载指定的图像和模板
    println!("加载图像{}...", screen_path);
    let screen_path = format!("images/{}", screen_path);
    let screen_image = image::open(&screen_path)?;

    println!("加载图像{}...", template_path);
    let template_path = format!("images/{}", template_path);
    let template_image = image::open(template_path)?;

    println!("加载图像完成");

    let (screen_width, screen_height) = screen_image.dimensions();

    println!("屏幕图像尺寸: {}x{}", screen_width, screen_height);

    let (template_width, template_height) = template_image.dimensions();

    println!("模板图像尺寸: {}x{}", template_width, template_height);

    let best_result = get_best_result(screen_path, template_width)?;

    println!("最佳匹配: ({},{})", best_result.x, best_result.y);

    start_match_with_try(
        screen_image.clone(),
        "FFT",
        || {
            ImageMatcher::new_from_image(
                template_image.clone(),
                MatcherMode::FFT {
                    src_width: screen_width,
                    src_height: screen_height,
                },
                None,
            )
        },
        best_result,
    );

    start_match_with_try(
        screen_image.clone(),
        "Segmented",
        || ImageMatcher::new_from_image(template_image.clone(), MatcherMode::Segmented, None),
        best_result,
    );

    Ok(())
}

fn start_match_with_try<F: FnOnce() -> ImageMatcher>(
    screen_img: DynamicImage,
    title: &str,
    matcher: F,
    best_result: MatchResult,
) {
    if let Err(err) = start_match(screen_img, title, matcher, best_result) {
        println!("{}匹配失败: {:?}", title, err);
    };
}
fn start_match<F: FnOnce() -> ImageMatcher>(
    screen_img: DynamicImage,
    title: &str,
    matcher: F,
    best_result: MatchResult,
) -> Result<(), Box<dyn Error>> {
    println!("{}", "-".repeat(120));

    let matcher = create_matcher(title, matcher);

    println!("开始{}匹配...", title);

    let start_time = time::Instant::now();

    let result = matcher.matching(screen_img, 0.9, Some(MatchHitFilter::new_default()))?;

    println!("{}匹配完成，耗时: {}", title, format_elapsed(start_time));

    let result_num = result.more_result.len() + 1;

    println!("找到 {} 个匹配:", result_num);
    println!(
        "最佳匹配: ({},{}),匹配度:{:.2?}",
        best_result.x, best_result.y, result.best_result.correlation
    );

    for hit in result.more_result.iter().take(5) {
        println!("({},{}) 匹配度:{:.2?}", hit.x, hit.y, hit.correlation);
    }

    if best_result.x != result.best_result.x || best_result.y != result.best_result.y {
        println!(
            "最佳匹配错误: ({},{})",
            result.best_result.x, result.best_result.y
        );
    }

    Ok(())
}

fn format_elapsed(start_time: time::Instant) -> String {
    let duration = start_time.elapsed();

    let secs = duration.as_secs();
    let mins = secs / 60;
    let secs = secs % 60;
    let millis = duration.as_millis() % 1000;

    let mut result = vec![];

    if mins > 0 {
        result.push(format!("{}m", mins));
    }

    if secs > 0 {
        result.push(format!("{}s", secs));
    }

    if millis > 0 {
        result.push(format!("{}ms", millis));
    }

    result.join("")
}
fn create_matcher<F: FnOnce() -> ImageMatcher>(title: &str, matcher: F) -> ImageMatcher {
    println!("开始创建{}匹配器...", title);
    let start_time = time::Instant::now();

    let matcher = matcher();

    println!(
        "{}匹配器创建完成，耗时: {}",
        title,
        format_elapsed(start_time)
    );

    matcher
}
