use image::open;
use image_matching_rs::{ImageMatcher, MatcherMode};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 图像匹配示例 (使用新的ImageMatcher API) ===");
    
    // 加载图像和模板
    println!("加载图像...");
    let screen_image = open("images/screen.png")?;
    let template_image = open("images/template.png")?.to_luma8();
    
    println!("屏幕图像尺寸: {}x{}", screen_image.width(), screen_image.height());
    println!("模板图像尺寸: {}x{}", template_image.width(), template_image.height());
    
    // 创建匹配器
    let mut matcher = ImageMatcher::new();
    
    // 准备模板（使用FFT模式）
    println!("准备模板 (FFT模式)...");
    matcher.prepare_template(
        &template_image,
        screen_image.width(),
        screen_image.height(),
        MatcherMode::FFT
    )?;
    
    // 执行匹配
    println!("执行匹配...");
    let start_time = std::time::Instant::now();
    let matches = matcher.matching(screen_image, MatcherMode::FFT, 0.8)?;
    let elapsed = start_time.elapsed();
    
    // 显示结果
    println!("匹配完成，耗时: {:.2}ms", elapsed.as_millis());
    println!("找到 {} 个匹配:", matches.len());
    
    for (i, result) in matches.iter().enumerate().take(10) {
        println!("  {}. 位置({}, {}), 尺寸: {}x{}, 相关系数: {:.3}", 
                 i + 1, result.x, result.y, result.width, result.height, result.correlation);
    }
    
    if matches.len() > 10 {
        println!("  ... 还有 {} 个匹配", matches.len() - 10);
    }
    
    // 显示最佳匹配
    if let Some(best_match) = matches.first() {
        println!("最佳匹配: 位置({}, {}), 相关系数: {:.3}", 
                 best_match.x, best_match.y, best_match.correlation);
    }
    
    // 演示分段模式（目前会返回错误，因为尚未实现）
    println!("\n=== 尝试分段模式 ===");
    match matcher.matching(open("images/screen.png")?, MatcherMode::Segmented, 0.8) {
        Ok(segmented_matches) => {
            println!("分段模式找到 {} 个匹配", segmented_matches.len());
        }
        Err(e) => {
            println!("分段模式: {}", e);
        }
    }
    
    Ok(())
}