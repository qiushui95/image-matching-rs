use image::{imageops, ImageBuffer, Rgba, RgbaImage};
use rand::Rng;
use serde::Serialize;
use std::fs::{create_dir_all, File};
use std::path::Path;

const TEMPLATE_SIZE: u32 = 200;
const SCREEN_W: u32 = 2560;
const SCREEN_H: u32 = 1440;

#[derive(Serialize)]
struct Placement {
    width: u32,
    x: u32,
    y: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = Path::new("images");
    create_dir_all(out_dir)?;

    // 1) 生成模板图 template.png
    let template = generate_colorful_template(TEMPLATE_SIZE, TEMPLATE_SIZE);
    let template_path = out_dir.join("template.png");
    template.save(&template_path)?;

    // 2) 生成屏幕图 screen.png，渐变背景 + 随机放置缩放模板图
    let mut screen = generate_gradient_background(SCREEN_W, SCREEN_H);

    let mut placements: Vec<Placement> = Vec::new();
    let mut rects: Vec<(u32, u32, u32, u32)> = Vec::new(); // (x, y, w, h)
    let mut rng = rand::thread_rng();

    let widths: Vec<u32> = (20..=TEMPLATE_SIZE).rev().step_by(10).collect(); // 200,190,...,20

    for &w in &widths {
        let scaled = imageops::resize(&template, w, w, imageops::FilterType::Lanczos3);

        // 尝试随机放置，避免重叠且不越界
        let max_x = SCREEN_W.saturating_sub(w);
        let max_y = SCREEN_H.saturating_sub(w);

        let mut placed = false;
        for _try in 0..100_000 {
            let x = rng.gen_range(0..=max_x);
            let y = rng.gen_range(0..=max_y);
            if !intersects_any(x, y, w, w, &rects) {
                imageops::overlay(&mut screen, &scaled, x as i64, y as i64);
                rects.push((x, y, w, w));
                placements.push(Placement { width: w, x, y });
                placed = true;
                break;
            }
        }

        if !placed {
            eprintln!(
                "Warning: failed to place template of width {} after many attempts",
                w
            );
        }
    }

    let screen_path = out_dir.join("screen.png");
    screen.save(&screen_path)?;

    // 3) 导出 screen.json，仅包含 width/x/y
    let json_path = out_dir.join("screen.json");
    let file = File::create(&json_path)?;
    serde_json::to_writer_pretty(file, &placements)?;

    // 额外：在控制台输出简要信息
    println!(
        "Generated:\n  {}\n  {}\n  {}",
        template_path.display(),
        screen_path.display(),
        json_path.display()
    );

    Ok(())
}

fn generate_colorful_template(w: u32, h: u32) -> RgbaImage {
    let mut img: RgbaImage = ImageBuffer::new(w, h);

    // 背景：双线性渐变，颜色随 x/y 变化
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

    // 叠加：四个角落色块，增强辨识度
    draw_rect(&mut img, 2, 2, 22, 22, Rgba([255, 0, 0, 255])); // 红色
    draw_rect(&mut img, w - 24, 2, 22, 22, Rgba([0, 255, 0, 255])); // 绿色
    draw_rect(&mut img, 2, h - 24, 22, 22, Rgba([0, 0, 255, 255])); // 蓝色
    draw_rect(&mut img, w - 24, h - 24, 22, 22, Rgba([255, 255, 0, 255])); // 黄色

    // 白色边框（2px）
    draw_border(&mut img, Rgba([255, 255, 255, 255]), 2);

    // 黑色对角线加粗（先画，保证中间图标覆盖在上层）
    draw_diagonal_x(&mut img, Rgba([0, 0, 0, 255]), 3);

    // 中心图标：靶心 + 十字准星，增强模板的中心特征
    let cx = (w / 2) as i32;
    let cy = (h / 2) as i32;
    // 同心圆（白-黑-红-白）
    draw_disk(&mut img, cx, cy, 64, Rgba([255, 255, 255, 255]));
    draw_disk(&mut img, cx, cy, 48, Rgba([0, 0, 0, 255]));
    draw_disk(&mut img, cx, cy, 32, Rgba([220, 20, 60, 255])); // Crimson red
    draw_disk(&mut img, cx, cy, 16, Rgba([255, 255, 255, 255]));
    // 十字准星
    draw_crosshair(&mut img, cx, cy, 70, 5, Rgba([0, 0, 0, 255]));

    img
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