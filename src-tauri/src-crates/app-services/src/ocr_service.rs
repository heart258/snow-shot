use std::path::{Path, PathBuf};

use num_cpus;
use rapid_ocr_rs::input::image_loader::OcrInput;
use rapid_ocr_rs::pipeline::config::EngineConfig;
use rapid_ocr_rs::pipeline::rapid_ocr::RapidOcr;
use rapid_ocr_rs::pipeline::types::OcrCallOptions;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize)]
pub struct OcrTextPoint {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct TextBlock {
    pub box_points: Vec<OcrTextPoint>,
    pub text: String,
    pub text_score: f32,
}

pub struct OcrService {
    hot_start: bool,
    ocr_core: Option<RapidOcr>,
    plugin_path: Option<PathBuf>,
    model: Option<OcrModel>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy, PartialOrd, Serialize, Deserialize)]
pub enum OcrModel {
    RapidOcrV4,
    RapidOcrV5,
    RapidOcrV6Tiny,
    RapidOcrV6Small,
    RapidOcrV6Medium,
}

/// 每个模型档位对应的模型文件与配置
struct ModelFiles {
    det_file: &'static str,
    det_version: &'static str,
    det_model_type: &'static str,
    cls_file: &'static str,
    rec_file: &'static str,
    rec_keys_file: &'static str,
    rec_version: &'static str,
    rec_model_type: &'static str,
}

impl OcrModel {
    fn files(self) -> ModelFiles {
        match self {
            OcrModel::RapidOcrV4 => ModelFiles {
                det_file: "ch_PP-OCRv4_det_infer.onnx",
                det_version: "PP-OCRv4",
                det_model_type: "mobile",
                cls_file: "ch_ppocr_mobile_v2.0_cls_infer.onnx",
                rec_file: "ch_PP-OCRv4_rec_infer.onnx",
                rec_keys_file: "ppocr_keys_v1.txt",
                rec_version: "PP-OCRv4",
                rec_model_type: "mobile",
            },
            OcrModel::RapidOcrV5 => ModelFiles {
                det_file: "ch_PP-OCRv5_mobile_det.onnx",
                det_version: "PP-OCRv5",
                det_model_type: "mobile",
                cls_file: "ch_ppocr_mobile_v2.0_cls_infer.onnx",
                rec_file: "ch_PP-OCRv5_rec_mobile_infer.onnx",
                rec_keys_file: "ppocrv5_dict.txt",
                rec_version: "PP-OCRv5",
                rec_model_type: "mobile",
            },
            OcrModel::RapidOcrV6Tiny => ModelFiles {
                det_file: "PP-OCRv6_det_tiny.onnx",
                det_version: "PP-OCRv6",
                det_model_type: "tiny",
                cls_file: "ch_ppocr_mobile_v2.0_cls_infer.onnx",
                rec_file: "PP-OCRv6_rec_tiny.onnx",
                rec_keys_file: "ppocrv6_tiny_dict.txt",
                rec_version: "PP-OCRv6",
                rec_model_type: "tiny",
            },
            OcrModel::RapidOcrV6Small => ModelFiles {
                det_file: "PP-OCRv6_det_small.onnx",
                det_version: "PP-OCRv6",
                det_model_type: "small",
                cls_file: "ch_ppocr_mobile_v2.0_cls_infer.onnx",
                rec_file: "PP-OCRv6_rec_small.onnx",
                rec_keys_file: "ppocrv6_dict.txt",
                rec_version: "PP-OCRv6",
                rec_model_type: "small",
            },
            OcrModel::RapidOcrV6Medium => ModelFiles {
                det_file: "PP-OCRv6_det_medium.onnx",
                det_version: "PP-OCRv6",
                det_model_type: "medium",
                cls_file: "ch_ppocr_mobile_v2.0_cls_infer.onnx",
                rec_file: "PP-OCRv6_rec_medium.onnx",
                rec_keys_file: "ppocrv6_dict.txt",
                rec_version: "PP-OCRv6",
                rec_model_type: "medium",
            },
        }
    }
}

/// 通过 YAML 字符串构造 EngineConfig，规避 paddle-ocr-rs 0.7.0 中私有模块类型的可见性问题。
fn build_engine_config(plugin_path: &Path, model: OcrModel) -> Result<EngineConfig, String> {
    let files = model.files();

    let det_path = plugin_path.join(files.det_file);
    let cls_path = plugin_path.join(files.cls_file);
    let rec_path = plugin_path.join(files.rec_file);
    let rec_keys_path = plugin_path.join(files.rec_keys_file);

    let yaml = format!(
        r#"
global:
  text_score: 0.5
  use_det: true
  use_cls: true
  use_rec: true
  min_height: 30
  width_height_ratio: 8.0
  max_side_len: 2000
  min_side_len: 30
  return_word_box: false
  return_single_char_box: false
det:
  model_path: {}
  allow_download: false
  ocr_version: {}
  model_type: {}
cls:
  model_path: {}
  allow_download: false
  ocr_version: PP-OCRv4
  model_type: mobile
rec:
  model:
    model_path: {}
    rec_keys_path: {}
    allow_download: false
    ocr_version: {}
    model_type: {}
"#,
        det_path.display(),
        files.det_version,
        files.det_model_type,
        cls_path.display(),
        rec_path.display(),
        rec_keys_path.display(),
        files.rec_version,
        files.rec_model_type,
    );

    EngineConfig::from_yaml_str(&yaml)
        .map_err(|e| format!("[OcrService::build_engine_config] Invalid engine config: {}", e))
}

fn convert_output(output: &rapid_ocr_rs::pipeline::types::OcrOutput) -> Vec<TextBlock> {
    let boxes = output.boxes.as_deref().unwrap_or(&[]);
    let txts = output.txts.as_deref().unwrap_or(&[]);
    let scores = output.scores.as_deref().unwrap_or(&[]);

    let n = boxes.len().min(txts.len()).min(scores.len());
    let mut blocks = Vec::with_capacity(n);

    for i in 0..n {
        let quad = boxes[i];
        let box_points = quad
            .iter()
            .map(|p| OcrTextPoint { x: p[0], y: p[1] })
            .collect();

        blocks.push(TextBlock {
            box_points,
            text: txts[i].clone(),
            text_score: scores[i],
        });
    }

    blocks
}

impl OcrService {
    pub fn new() -> Self {
        Self {
            hot_start: false,
            ocr_core: None,
            plugin_path: None,
            model: None,
        }
    }

    pub async fn init_models(
        &mut self,
        orc_plugin_path: PathBuf,
        model: OcrModel,
        hot_start: bool,
        _ocr_model_write_to_memory: bool,
    ) -> Result<(), String> {
        log::info!(
            "[OcrService::init_models] orc_plugin_path: {:?}, model: {:?}, hot_start: {:?}",
            orc_plugin_path,
            model,
            hot_start
        );

        self.plugin_path = Some(orc_plugin_path);
        self.model = Some(model);
        self.hot_start = hot_start;

        if self.hot_start {
            self.init_session().await?;
        } else {
            self.ocr_core.take();
        }

        Ok(())
    }

    pub async fn init_session(&mut self) -> Result<(), String> {
        let plugin_path = self
            .plugin_path
            .clone()
            .ok_or("[OcrService::init_session] Plugin path is not loaded")?;
        let model = self
            .model
            .ok_or("[OcrService::init_session] Model is not loaded")?;

        let config = build_engine_config(&plugin_path, model)?;
        let ocr_core = RapidOcr::new(config)
            .map_err(|e| format!("[OcrService::init_session] Failed to init models: {}", e))?;

        self.ocr_core.replace(ocr_core);

        Ok(())
    }

    /// 释放 onnx session，并初始化新的 session
    pub async fn release_session(&mut self) -> Result<(), String> {
        if self.hot_start {
            self.init_session().await?;
        } else {
            self.ocr_core.take();
        }

        Ok(())
    }

    pub async fn get_session(&mut self) -> Result<&mut RapidOcr, String> {
        if self.ocr_core.is_none() {
            self.init_session().await?;
        }

        Ok(self.ocr_core.as_mut().unwrap())
    }

    /// 对 RGB 图像执行 OCR，返回与旧版 TextBlock 兼容的结构。
    pub async fn run_ocr(
        &mut self,
        width: usize,
        height: usize,
        rgb_data: &[u8],
        detect_angle: bool,
    ) -> Result<Vec<TextBlock>, String> {
        let ocr_core = self.get_session().await?;

        let input = OcrInput::RgbU8 {
            width,
            height,
            data: rgb_data.to_vec(),
        };

        let opts = OcrCallOptions {
            use_det: Some(true),
            use_cls: Some(detect_angle),
            use_rec: Some(true),
            ..Default::default()
        };

        let output = ocr_core
            .run(input, opts)
            .map_err(|e| format!("[OcrService::run_ocr] Failed to run OCR: {}", e))?;

        Ok(convert_output(&output))
    }
}