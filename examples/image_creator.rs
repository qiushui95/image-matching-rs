use image::{imageops, ImageBuffer, Rgba, RgbaImage};
use rand::Rng;
use serde::Serialize;
use std::fs::{create_dir_all, File};
use std::path::Path;

#[derive(Serialize)]
struct Placement {
    width: u32,
    x: u32,
    y: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = Path::new("images");
    create_dir_all(out_dir)?;

    // 定义4个模板尺寸和3个屏幕分辨率
    let template_sizes = vec![50, 100, 150, 200];
    let screen_sizes = vec![(1024, 768), (1920, 1080), (2560, 1440)];

    // 1) 生成4个不同的模板图
    let mut templates = Vec::new();
    for &template_size in &template_sizes {
        let template = generate_colorful_template(template_size, template_size);
        let template_filename = format!("template{}x{}.png", template_size, template_size);
        let template_path = out_dir.join(&template_filename);
        template.save(&template_path)?;
        println!("Generated: {}", template_path.display());
        templates.push((template_size, template));
    }

    // 2) 生成3个目标图，每个包含所有4种模板的缩放实例
    for &(screen_w, screen_h) in &screen_sizes {
        let mut screen = generate_gradient_background(screen_w, screen_h);
        let mut placements: Vec<Placement> = Vec::new();
        let mut rects: Vec<(u32, u32, u32, u32)> = Vec::new(); // (x, y, w, h)
        let mut rng = rand::thread_rng();

        // 缩放比例：0%, 20%, 40%, 60%, 80%（即保留100%, 80%, 60%, 40%, 20%的尺寸）
        let scale_factors = vec![1.0, 0.8, 0.6, 0.4, 0.2];

        // 为每个模板的每个缩放比例放置实例
        for (template_size, template) in &templates {
            for &scale in &scale_factors {
                let scaled_size = (*template_size as f32 * scale) as u32;
                if scaled_size < 5 { continue; } // 跳过太小的尺寸
                
                let scaled = imageops::resize(template, scaled_size, scaled_size, imageops::FilterType::Lanczos3);

                // 尝试随机放置，避免重叠且不越界
                let max_x = screen_w.saturating_sub(scaled_size);
                let max_y = screen_h.saturating_sub(scaled_size);

                let mut placed = false;
                for _try in 0..100_000 {
                    let x = rng.gen_range(0..=max_x);
                    let y = rng.gen_range(0..=max_y);

                    if !intersects_any(x, y, scaled_size, scaled_size, &rects) {
                        // 叠加到屏幕图上
                        imageops::overlay(&mut screen, &scaled, x as i64, y as i64);
                        placements.push(Placement { 
                            width: scaled_size, 
                            x, 
                            y 
                        });
                        rects.push((x, y, scaled_size, scaled_size));
                        placed = true;
                        break;
                    }
                }

                if !placed {
                    eprintln!(
                        "Warning: failed to place template {}x{} with scale {:.1} (size {}) after many attempts",
                        template_size, template_size, scale, scaled_size
                    );
                }
            }
        }

        // 保存屏幕图
        let screen_filename = format!("screen{}x{}.png", screen_w, screen_h);
        let screen_path = out_dir.join(&screen_filename);
        screen.save(&screen_path)?;
        println!("Generated: {}", screen_path.display());

        // 3) 导出 JSON，仅包含 width/x/y
        let json_filename = format!("screen{}x{}.json", screen_w, screen_h);
        let json_path = out_dir.join(&json_filename);
        let file = File::create(&json_path)?;
        serde_json::to_writer_pretty(file, &placements)?;
        println!("Generated: {}", json_path.display());
    }

    Ok(())
}

fn generate_colorful_template(w: u32, h: u32) -> RgbaImage {
    let mut img: RgbaImage = ImageBuffer::new(w, h);

    // 根据尺寸选择不同的设计方案
    match w {
        50 => generate_template_50x50(&mut img, w, h),
        100 => generate_template_100x100(&mut img, w, h),
        150 => generate_template_150x150(&mut img, w, h),
        200 => generate_template_200x200(&mut img, w, h),
        _ => generate_template_default(&mut img, w, h),
    }

    img
}

// 200x200 模板：靶心设计
fn generate_template_200x200(img: &mut RgbaImage, w: u32, h: u32) {
    // 背景：蓝色到紫色渐变
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 / (w - 1) as f32;
            let fy = y as f32 / (h - 1) as f32;
            let r = (fx * 100.0 + 50.0) as u8;
            let g = (fy * 80.0 + 30.0) as u8;
            let b = (200.0 + fx * 55.0) as u8;
            img.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }

    // 四个角落色块 - 不同颜色组合
    let corner_size = (w / 10).max(4);
    let corner_margin = (w / 50).max(1);
    draw_rect(img, corner_margin, corner_margin, corner_size, corner_size, Rgba([255, 100, 100, 255])); // 粉红
    draw_rect(img, w - corner_margin - corner_size, corner_margin, corner_size, corner_size, Rgba([100, 255, 100, 255])); // 浅绿
    draw_rect(img, corner_margin, h - corner_margin - corner_size, corner_size, corner_size, Rgba([100, 100, 255, 255])); // 浅蓝
    draw_rect(img, w - corner_margin - corner_size, h - corner_margin - corner_size, corner_size, corner_size, Rgba([255, 255, 100, 255])); // 浅黄

    // 金色边框
    let border_thickness = (w / 100).max(1);
    draw_border(img, Rgba([255, 215, 0, 255]), border_thickness);

    // 白色对角线
    let diagonal_thickness = (w / 67).max(1);
    draw_diagonal_x(img, Rgba([255, 255, 255, 255]), diagonal_thickness);

    // 中心靶心设计
    let cx = (w / 2) as i32;
    let cy = (h / 2) as i32;
    let base_radius = (w / 6).max(8);
    
    // 同心圆（白-黑-红-白）
    draw_disk(img, cx, cy, (base_radius * 4 / 5) as i32, Rgba([255, 255, 255, 255]));
    draw_disk(img, cx, cy, (base_radius * 3 / 5) as i32, Rgba([0, 0, 0, 255]));
    draw_disk(img, cx, cy, (base_radius * 2 / 5) as i32, Rgba([220, 20, 60, 255]));
    draw_disk(img, cx, cy, (base_radius / 5) as i32, Rgba([255, 255, 255, 255]));
    
    // 十字准星
    let crosshair_size = (base_radius * 7 / 4) as i32;
    let crosshair_thickness = (w / 40).max(1);
    draw_crosshair(img, cx, cy, crosshair_size, crosshair_thickness, Rgba([0, 0, 0, 255]));
}

// 100x100 模板：钻石设计
fn generate_template_100x100(img: &mut RgbaImage, w: u32, h: u32) {
    // 背景：紫色到粉色渐变
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 / (w - 1) as f32;
            let fy = y as f32 / (h - 1) as f32;
            let r = (180.0 + fx * 75.0) as u8;
            let g = (50.0 + fy * 150.0) as u8;
            let b = (200.0 - fx * 50.0) as u8;
            img.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }

    // 四个角落色块 - 金属色调
    let corner_size = (w / 8).max(4);
    let corner_margin = (w / 25).max(1);
    draw_rect(img, corner_margin, corner_margin, corner_size, corner_size, Rgba([255, 215, 0, 255])); // 金色
    draw_rect(img, w - corner_margin - corner_size, corner_margin, corner_size, corner_size, Rgba([192, 192, 192, 255])); // 银色
    draw_rect(img, corner_margin, h - corner_margin - corner_size, corner_size, corner_size, Rgba([184, 115, 51, 255])); // 铜色
    draw_rect(img, w - corner_margin - corner_size, h - corner_margin - corner_size, corner_size, corner_size, Rgba([229, 228, 226, 255])); // 铂金色

    // 紫色边框
    let border_thickness = (w / 50).max(1);
    draw_border(img, Rgba([128, 0, 128, 255]), border_thickness);

    // 金色对角线
    let diagonal_thickness = (w / 33).max(1);
    draw_diagonal_x(img, Rgba([255, 215, 0, 255]), diagonal_thickness);

    // 中心钻石设计
    let cx = (w / 2) as i32;
    let cy = (h / 2) as i32;
    let diamond_size = (w / 4).max(8);
    
    // 绘制钻石形状（菱形）
    draw_diamond(img, cx, cy, diamond_size as i32, Rgba([255, 255, 255, 255]));
    draw_diamond(img, cx, cy, (diamond_size * 3 / 4) as i32, Rgba([0, 0, 0, 255]));
    draw_diamond(img, cx, cy, (diamond_size / 2) as i32, Rgba([255, 20, 147, 255])); // 深粉色
    draw_diamond(img, cx, cy, (diamond_size / 4) as i32, Rgba([255, 255, 255, 255]));
}

// 150x150 模板：六边形设计
fn generate_template_150x150(img: &mut RgbaImage, w: u32, h: u32) {
    // 背景：绿色到蓝色渐变
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 / (w - 1) as f32;
            let fy = y as f32 / (h - 1) as f32;
            let r = (20.0 + fx * 80.0) as u8;
            let g = (100.0 + fy * 155.0) as u8;
            let b = (150.0 + fx * 105.0) as u8;
            img.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }

    // 四个角落色块 - 彩虹色调
    let corner_size = (w / 9).max(4);
    let corner_margin = (w / 30).max(1);
    draw_rect(img, corner_margin, corner_margin, corner_size, corner_size, Rgba([255, 69, 0, 255])); // 橙红色
    draw_rect(img, w - corner_margin - corner_size, corner_margin, corner_size, corner_size, Rgba([50, 205, 50, 255])); // 酸橙绿
    draw_rect(img, corner_margin, h - corner_margin - corner_size, corner_size, corner_size, Rgba([138, 43, 226, 255])); // 蓝紫色
    draw_rect(img, w - corner_margin - corner_size, h - corner_margin - corner_size, corner_size, corner_size, Rgba([255, 20, 147, 255])); // 深粉色

    // 青色边框
    let border_thickness = (w / 75).max(1);
    draw_border(img, Rgba([0, 255, 255, 255]), border_thickness);

    // 红色对角线
    let diagonal_thickness = (w / 50).max(1);
    draw_diagonal_x(img, Rgba([255, 0, 0, 255]), diagonal_thickness);

    // 中心六边形设计
    let cx = (w / 2) as i32;
    let cy = (h / 2) as i32;
    let hex_size = (w / 5).max(10);
    
    // 绘制同心六边形
    draw_hexagon(img, cx, cy, hex_size as i32, Rgba([255, 255, 255, 255]));
    draw_hexagon(img, cx, cy, (hex_size * 4 / 5) as i32, Rgba([0, 0, 0, 255]));
    draw_hexagon(img, cx, cy, (hex_size * 3 / 5) as i32, Rgba([0, 255, 127, 255])); // 春绿色
    draw_hexagon(img, cx, cy, (hex_size * 2 / 5) as i32, Rgba([255, 255, 255, 255]));
    draw_hexagon(img, cx, cy, (hex_size / 5) as i32, Rgba([255, 140, 0, 255])); // 深橙色
}

// 50x50 模板：星形设计
fn generate_template_50x50(img: &mut RgbaImage, w: u32, h: u32) {
    // 背景：橙色到红色渐变
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 / (w - 1) as f32;
            let fy = y as f32 / (h - 1) as f32;
            let r = (255.0 - fx * 50.0) as u8;
            let g = (150.0 - fy * 100.0) as u8;
            let b = (fx * fy * 100.0) as u8;
            img.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }

    // 四个角落色块 - 对比色
    let corner_size = (w / 8).max(3);
    let corner_margin = (w / 25).max(1);
    draw_rect(img, corner_margin, corner_margin, corner_size, corner_size, Rgba([0, 255, 255, 255])); // 青色
    draw_rect(img, w - corner_margin - corner_size, corner_margin, corner_size, corner_size, Rgba([255, 0, 255, 255])); // 洋红
    draw_rect(img, corner_margin, h - corner_margin - corner_size, corner_size, corner_size, Rgba([255, 255, 0, 255])); // 黄色
    draw_rect(img, w - corner_margin - corner_size, h - corner_margin - corner_size, corner_size, corner_size, Rgba([0, 0, 0, 255])); // 黑色

    // 蓝色边框
    let border_thickness = (w / 50).max(1);
    draw_border(img, Rgba([0, 100, 255, 255]), border_thickness);

    // 绿色对角线
    let diagonal_thickness = (w / 25).max(1);
    draw_diagonal_x(img, Rgba([0, 255, 0, 255]), diagonal_thickness);

    // 中心星形设计
    let cx = (w / 2) as i32;
    let cy = (h / 2) as i32;
    let star_size = (w / 4).max(6);
    
    // 绘制星形（使用多个三角形近似）
    draw_star(img, cx, cy, star_size as i32, Rgba([255, 255, 255, 255]));
    draw_star(img, cx, cy, (star_size * 2 / 3) as i32, Rgba([0, 0, 0, 255]));
    draw_star(img, cx, cy, (star_size / 3) as i32, Rgba([255, 215, 0, 255])); // 金色中心
}

// 默认模板设计
fn generate_template_default(img: &mut RgbaImage, w: u32, h: u32) {
    // 背景：双线性渐变
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 / (w - 1) as f32;
            let fy = y as f32 / (h - 1) as f32;
            let r = (fx * 255.0) as u8;
            let g = (fy * 255.0) as u8;
            let b = (((fx + fy) / 2.0) * 255.0) as u8;
            img.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }

    // 标准设计
    let corner_size = (w / 10).max(4);
    let corner_margin = (w / 50).max(1);
    draw_rect(img, corner_margin, corner_margin, corner_size, corner_size, Rgba([255, 0, 0, 255]));
    draw_rect(img, w - corner_margin - corner_size, corner_margin, corner_size, corner_size, Rgba([0, 255, 0, 255]));
    draw_rect(img, corner_margin, h - corner_margin - corner_size, corner_size, corner_size, Rgba([0, 0, 255, 255]));
    draw_rect(img, w - corner_margin - corner_size, h - corner_margin - corner_size, corner_size, corner_size, Rgba([255, 255, 0, 255]));

    let border_thickness = (w / 100).max(1);
    draw_border(img, Rgba([255, 255, 255, 255]), border_thickness);

    let diagonal_thickness = (w / 67).max(1);
    draw_diagonal_x(img, Rgba([0, 0, 0, 255]), diagonal_thickness);

    let cx = (w / 2) as i32;
    let cy = (h / 2) as i32;
    let base_radius = (w / 6).max(8);
    
    draw_disk(img, cx, cy, (base_radius * 4 / 5) as i32, Rgba([255, 255, 255, 255]));
    draw_disk(img, cx, cy, (base_radius * 3 / 5) as i32, Rgba([0, 0, 0, 255]));
    draw_disk(img, cx, cy, (base_radius * 2 / 5) as i32, Rgba([220, 20, 60, 255]));
    draw_disk(img, cx, cy, (base_radius / 5) as i32, Rgba([255, 255, 255, 255]));
    
    let crosshair_size = (base_radius * 7 / 4) as i32;
    let crosshair_thickness = (w / 40).max(1);
    draw_crosshair(img, cx, cy, crosshair_size, crosshair_thickness, Rgba([0, 0, 0, 255]));
}

fn generate_gradient_background(w: u32, h: u32) -> RgbaImage {
    let mut img: RgbaImage = ImageBuffer::new(w, h);
    // 垂直渐变：深蓝到青绿
    let start = [30u8, 50u8, 120u8];
    let end = [30u8, 180u8, 200u8];

    for y in 0..h {
        let t = y as f32 / (h - 1) as f32;
        let r = lerp_u8(start[0], end[0], t);
        let g = lerp_u8(start[1], end[1], t);
        let b = lerp_u8(start[2], end[2], t);
        for x in 0..w {
            img.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }
    img
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    let av = a as f32;
    let bv = b as f32;
    ((av * (1.0 - t) + bv * t).clamp(0.0, 255.0)) as u8
}

fn draw_rect(img: &mut RgbaImage, x: u32, y: u32, w: u32, h: u32, color: Rgba<u8>) {
    let max_x = img.width();
    let max_y = img.height();
    for yy in y..y.saturating_add(h).min(max_y) {
        for xx in x..x.saturating_add(w).min(max_x) {
            img.put_pixel(xx, yy, color);
        }
    }
}

fn draw_border(img: &mut RgbaImage, color: Rgba<u8>, thickness: u32) {
    let w = img.width();
    let h = img.height();
    for t in 0..thickness {
        // 顶部和底部
        for x in 0..w {
            img.put_pixel(x, t.min(h - 1), color);
            img.put_pixel(x, h.saturating_sub(1 + t), color);
        }
        // 左右
        for y in 0..h {
            img.put_pixel(t.min(w - 1), y, color);
            img.put_pixel(w.saturating_sub(1 + t), y, color);
        }
    }
}

fn draw_diagonal_x(img: &mut RgbaImage, color: Rgba<u8>, thickness: u32) {
    let w = img.width();
    let h = img.height();
    let t = thickness as i32;
    for y in 0..h {
        for x in 0..w {
            let xi = x as i32;
            let yi = y as i32;
            let d1 = (yi - xi).abs();
            let d2 = (yi - (h as i32 - 1 - xi)).abs();
            if d1 <= t / 2 || d2 <= t / 2 {
                img.put_pixel(x, y, color);
            }
        }
    }
}

fn draw_disk(img: &mut RgbaImage, cx: i32, cy: i32, r: i32, color: Rgba<u8>) {
    if r <= 0 { return; }
    let w = img.width() as i32;
    let h = img.height() as i32;
    let r2 = r * r;
    for dy in -r..=r {
        let yy = cy + dy;
        if yy < 0 || yy >= h { continue; }
        for dx in -r..=r {
            let xx = cx + dx;
            if xx < 0 || xx >= w { continue; }
            if dx*dx + dy*dy <= r2 {
                img.put_pixel(xx as u32, yy as u32, color);
            }
        }
    }
}

fn draw_crosshair(img: &mut RgbaImage, cx: i32, cy: i32, size: i32, thickness: u32, color: Rgba<u8>) {
    let w = img.width() as i32;
    let h = img.height() as i32;
    let half = size / 2;
    let t = thickness as i32;
    // 水平线
    for dy in -t/2..=t/2 {
        let yy = cy + dy;
        if yy < 0 || yy >= h { continue; }
        for xx in (cx - half).max(0)..=(cx + half).min(w - 1) {
            img.put_pixel(xx as u32, yy as u32, color);
        }
    }
    // 垂直线
    for dx in -t/2..=t/2 {
        let xx = cx + dx;
        if xx < 0 || xx >= w { continue; }
        for yy in (cy - half).max(0)..=(cy + half).min(h - 1) {
            img.put_pixel(xx as u32, yy as u32, color);
        }
    }
}

fn draw_star(img: &mut RgbaImage, cx: i32, cy: i32, size: i32, color: Rgba<u8>) {
    // 绘制五角星，使用简化的方法：绘制多条线段
    let outer_radius = size;
    let inner_radius = size * 2 / 5;
    
    // 五角星的10个顶点（5个外顶点 + 5个内顶点）
    let mut points = Vec::new();
    for i in 0..10 {
        let angle = (i as f32 * std::f32::consts::PI / 5.0) - std::f32::consts::PI / 2.0;
        let radius = if i % 2 == 0 { outer_radius } else { inner_radius } as f32;
        let x = cx + (radius * angle.cos()) as i32;
        let y = cy + (radius * angle.sin()) as i32;
        points.push((x, y));
    }
    
    // 连接相邻的点形成星形
    for i in 0..10 {
        let (x1, y1) = points[i];
        let (x2, y2) = points[(i + 1) % 10];
        draw_line(img, x1, y1, x2, y2, color);
    }
    
    // 填充中心区域
    draw_disk(img, cx, cy, inner_radius / 2, color);
}

fn draw_diamond(img: &mut RgbaImage, cx: i32, cy: i32, size: i32, color: Rgba<u8>) {
    // 绘制钻石形状（菱形）
    let w = img.width() as i32;
    let h = img.height() as i32;
    
    for dy in -size..=size {
        let yy = cy + dy;
        if yy < 0 || yy >= h { continue; }
        
        let width = size - dy.abs();
        for dx in -width..=width {
            let xx = cx + dx;
            if xx < 0 || xx >= w { continue; }
            img.put_pixel(xx as u32, yy as u32, color);
        }
    }
}

fn draw_hexagon(img: &mut RgbaImage, cx: i32, cy: i32, size: i32, color: Rgba<u8>) {
    // 绘制六边形
    let mut points = Vec::new();
    for i in 0..6 {
        let angle = (i as f32 * std::f32::consts::PI / 3.0);
        let x = cx + (size as f32 * angle.cos()) as i32;
        let y = cy + (size as f32 * angle.sin()) as i32;
        points.push((x, y));
    }
    
    // 连接相邻的点形成六边形
    for i in 0..6 {
        let (x1, y1) = points[i];
        let (x2, y2) = points[(i + 1) % 6];
        draw_line(img, x1, y1, x2, y2, color);
    }
    
    // 简单填充中心区域
    draw_disk(img, cx, cy, size * 2 / 3, color);
}

fn draw_line(img: &mut RgbaImage, x1: i32, y1: i32, x2: i32, y2: i32, color: Rgba<u8>) {
    // 简单的线段绘制算法（Bresenham算法的简化版）
    let dx = (x2 - x1).abs();
    let dy = (y2 - y1).abs();
    let sx = if x1 < x2 { 1 } else { -1 };
    let sy = if y1 < y2 { 1 } else { -1 };
    let mut err = dx - dy;
    let mut x = x1;
    let mut y = y1;
    
    loop {
        if x >= 0 && x < img.width() as i32 && y >= 0 && y < img.height() as i32 {
            img.put_pixel(x as u32, y as u32, color);
        }
        
        if x == x2 && y == y2 {
            break;
        }
        
        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
    }
}

fn intersects_any(x: u32, y: u32, w: u32, h: u32, rects: &[(u32, u32, u32, u32)]) -> bool {
    for &(rx, ry, rw, rh) in rects {
        if rects_intersect(x, y, w, h, rx, ry, rw, rh) {
            return true;
        }
    }
    false
}

fn rects_intersect(x1: u32, y1: u32, w1: u32, h1: u32, x2: u32, y2: u32, w2: u32, h2: u32) -> bool {
    let left1 = x1;
    let right1 = x1.saturating_add(w1);
    let top1 = y1;
    let bottom1 = y1.saturating_add(h1);

    let left2 = x2;
    let right2 = x2.saturating_add(w2);
    let top2 = y2;
    let bottom2 = y2.saturating_add(h2);

    !(right1 <= left2 || right2 <= left1 || bottom1 <= top2 || bottom2 <= top1)
}