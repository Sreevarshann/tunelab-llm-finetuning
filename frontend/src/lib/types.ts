export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export interface SplitMeta { count: number; sha256: string; tokens: { mean?: number; p95?: number; max?: number; total?: number } }
export interface Dataset {
  id: string; name: string; task: string; source: string; synthetic: boolean;
  scoring: { mode: "json_fields"; fields: string[] } | { mode: "text" };
  splits: Record<string, SplitMeta>; created_at: string;
}
export interface LoraParams {
  rank: number; scale: number; dropout: number; num_layers: number; learning_rate: number; iters: number;
  batch_size: number; max_seq_length: number; steps_per_report: number; steps_per_eval: number; seed: number; mask_prompt: boolean;
}
export interface CurvePoint { iteration: number; loss: number }
export interface TrainResult {
  adapter_dir: string; adapter_sha256: string; trainable_params: number; total_params: number;
  lora_b_norm_before: number; lora_b_norm_after: number; initial_val_loss: number; final_val_loss: number;
  final_train_loss: number | null; duration_s: number; peak_memory_gb: number;
  train_curve: (CurvePoint & { lr: number; tokens_per_second: number; peak_memory_gb: number })[]; val_curve: CurvePoint[];
}
export interface JsonMetrics { n: number; json_valid_pct: number; schema_exact_pct: number; all_fields_correct_pct: number; field_accuracy_pct: Record<string, number>; loss: number }
export interface TextMetrics { n: number; exact_match_pct: number; token_f1_pct: number; loss: number }
export type Metrics = JsonMetrics | TextMetrics;
export interface EvalResult { metrics: Metrics; duration_s: number; tokens_per_s: number | null; examples_file: string }
export interface Job<R = unknown> {
  id: string; kind: "train" | "evaluate"; name: string; dataset_id: string; status: JobStatus; config: Partial<LoraParams> & Record<string, unknown>;
  result: R | null; error: string | null; split: string | null; adapter_job_id: string | null; progress: number;
  created_at: string; started_at: string | null; finished_at: string | null;
}
export type Run = Job<TrainResult>;
export type Evaluation = Job<EvalResult>;
export interface RunDetail extends Run { dataset: Dataset; evaluations: Evaluation[] }
export interface Example {
  index: number; input: string; reference: string; output: string;
  parsed?: boolean; fields?: Record<string, boolean>; all_correct?: boolean; exact?: boolean; f1?: number;
}
export const isJson = (m: Metrics): m is JsonMetrics => "all_fields_correct_pct" in m;
