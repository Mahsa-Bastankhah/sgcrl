"""Read all PPO eval CSVs and write a self-contained canvas.tsx to the
Cursor canvases folder.  Run periodically to keep the canvas live.

Usage:
    python gen_training_canvas.py
"""
import csv, json, math, os, textwrap, sys
from datetime import datetime, timezone

CANVAS_PATH = (
    "/u/mb6458/.cursor/projects/n-fs-mislresearch-sgcrl/"
    "canvases/ppo-training-monitor.canvas.tsx"
)
LOG_BASE = "/n/fs/mislresearch/sgcrl/logs/ppo"
LOG_BASES = {
    # ablation runs live under different roots
    "btn_noanneal":         "/n/fs/mislresearch/sgcrl/logs/ppo_btn_noanneal",
    "btn_noanneal_lowent":  "/n/fs/mislresearch/sgcrl/logs/ppo_btn_noanneal_lowent",
}

# ── helpers ────────────────────────────────────────────────────────────────

def read_csv(env, seed, log_type="eval", log_base=None):
    base = log_base if log_base else LOG_BASE
    path = os.path.join(base, f"ppo_{env}_{seed}", "logs", log_type, "logs.csv")  # env may be overridden via env_dir
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def coerce(v, fallback=None):
    try:
        x = float(v)
        return None if (math.isnan(x) or math.isinf(x)) else x
    except Exception:
        return fallback


def extract(rows, x_col, y_col):
    pts = []
    for r in rows:
        x = coerce(r.get(x_col, ""))
        y = coerce(r.get(y_col, ""))
        if x is not None and y is not None:
            pts.append((int(x), y))
    return pts


def subsample(pts, max_pts=120):
    if len(pts) <= max_pts:
        return pts
    step = max(1, len(pts) // max_pts)
    sampled = pts[::step]
    if pts[-1] not in sampled:
        sampled = sampled + [pts[-1]]
    return sampled


def align_seeds(seed_pts, max_pts=120):
    """Return (categories, per_seed_values) with None for missing x values."""
    all_x = sorted({x for pts in seed_pts.values() for x, _ in pts})
    if not all_x:
        return [], {}
    # downsample categories if needed
    if len(all_x) > max_pts:
        step = max(1, len(all_x) // max_pts)
        all_x_set = set(all_x[::step]) | {all_x[-1]}
        all_x = sorted(all_x_set)

    cats = [str(x) for x in all_x]
    per_seed = {}
    for s, pts in seed_pts.items():
        lookup = {x: y for x, y in pts}
        vals = [lookup.get(x) for x in all_x]
        per_seed[s] = vals
    return cats, per_seed


def compute_mean_std(per_seed, cats_len):
    """Return (mean_list, std_list) over available seeds at each x."""
    mean_list, std_list = [], []
    for i in range(cats_len):
        vals = [v for v in [per_seed[s][i] for s in per_seed] if v is not None]
        if vals:
            m = sum(vals) / len(vals)
            std = (
                math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
                if len(vals) > 1 else 0.0
            )
        else:
            m, std = None, None
        mean_list.append(m)
        std_list.append(std)
    return mean_list, std_list


def fill_none_with_prev(lst):
    """Forward-fill None values for smoother lines."""
    out, prev = [], None
    for v in lst:
        if v is not None:
            prev = v
        out.append(prev)
    return out

# ── data collection ────────────────────────────────────────────────────────

FLOW_HORIZON = 1500  # episode length for all figure-eight variants

ENVS = {
    "sawyer_push":              dict(label="Sawyer Push",              metric="success_1000", ylabel="Success ×1000"),
    "sawyer_drawer_open":       dict(label="Sawyer Drawer Open",       metric="success_1000", ylabel="Success ×1000"),
    "sawyer_button_press":      dict(label="Sawyer Button Press",      metric="success_1000", ylabel="Success ×1000"),
    "flow_figureeight":         dict(label="Flow Figure-Eight (1/14)", metric="ep_flow_dense_return_mean", ylabel="Episode Return",  log="learner"),
    "flow_figureeight_7rl":     dict(label="Flow Figure-Eight (7/14)", metric="ep_flow_dense_return_mean", ylabel="Episode Return",  log="learner"),
    "flow_figureeight_14rl":    dict(label="Flow Figure-Eight (14/14)",metric="ep_flow_dense_return_mean", ylabel="Episode Return",  log="learner"),
    # Success rate = fraction of steps where ALL vehicles are within 1 m/s of target.
    # ep_return_mean is the sparse 0/1 reward summed over the 1500-step episode;
    # divide by FLOW_HORIZON to get a [0, 1] success fraction.
    "flow_fe7rl_success":  dict(label="Flow 7/14 – Success Rate",  env_dir="flow_figureeight_7rl",  metric="ep_return_mean", scale=1/FLOW_HORIZON, ylabel="Success Rate", log="learner"),
    "flow_fe14rl_success": dict(label="Flow 14/14 – Success Rate", env_dir="flow_figureeight_14rl", metric="ep_return_mean", scale=1/FLOW_HORIZON, ylabel="Success Rate", log="learner"),
    # 2-vehicle variants
    "flow_figureeight_2v1rl": dict(label="Flow 2-Car (1/2 RL)", metric="ep_flow_dense_return_mean", ylabel="Episode Return", log="learner"),
    "flow_figureeight_2v2rl": dict(label="Flow 2-Car (2/2 RL)", metric="ep_flow_dense_return_mean", ylabel="Episode Return", log="learner"),
    # 1-vehicle sanity check
    "flow_figureeight_1v1rl": dict(label="Flow 1-Car Sanity (1/1 RL)", metric="ep_flow_dense_return_mean", ylabel="Episode Return", log="learner"),
    # scaling experiments
    "flow_figureeight_4v2rl": dict(label="Flow 4-Car (2/4 RL)", metric="ep_flow_dense_return_mean", ylabel="Episode Return", log="learner"),
    "flow_figureeight_8v4rl": dict(label="Flow 8-Car (4/8 RL)", metric="ep_flow_dense_return_mean", ylabel="Episode Return", log="learner"),
    # button-press ablations
    "btn_noanneal":        dict(label="Button (no LR sched, ent=0.05)",
                                env_dir="sawyer_button_press",
                                metric="success_1000", ylabel="Success ×1000",
                                log_base="btn_noanneal"),
    "btn_noanneal_lowent": dict(label="Button (no LR sched, ent=0.01)",
                                env_dir="sawyer_button_press",
                                metric="success_1000", ylabel="Success ×1000",
                                log_base="btn_noanneal_lowent"),
}

charts_data = {}
for env, cfg in ENVS.items():
    log_type = cfg.get("log", "eval")
    env_dir = cfg.get("env_dir", env)  # directory name may differ from the ENVS key
    scale = cfg.get("scale", 1.0)
    log_base_key = cfg.get("log_base")
    log_base_dir = LOG_BASES.get(log_base_key) if log_base_key else None
    x_col = "learner_steps"
    seed_pts = {}
    for s in range(3):
        rows = read_csv(env_dir, s, log_type, log_base=log_base_dir)
        pts = extract(rows, x_col, cfg["metric"])
        if scale != 1.0:
            pts = [(x, y * scale) for x, y in pts]
        if pts:
            seed_pts[s] = subsample(pts, max_pts=120)

    cats, per_seed = align_seeds(seed_pts, max_pts=120)
    if cats:
        mean_vals, std_vals = compute_mean_std(per_seed, len(cats))
        upper = [
            m + sd if m is not None and sd is not None else m
            for m, sd in zip(mean_vals, std_vals)
        ]
        lower = [
            max(0, m - sd) if m is not None and sd is not None else m
            for m, sd in zip(mean_vals, std_vals)
        ]
        charts_data[env] = {
            "label": cfg["label"],
            "ylabel": cfg["ylabel"],
            "cats": cats,
            "mean": fill_none_with_prev(mean_vals),
            "upper": fill_none_with_prev(upper),
            "lower": fill_none_with_prev(lower),
        }

ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# ── canvas template ────────────────────────────────────────────────────────

CANVAS = textwrap.dedent(f"""\
import {{ Stack, Grid, Card, CardHeader, CardBody,
         H1, H2, Text, Row, useHostTheme }} from "cursor/canvas";

const CHARTS = {json.dumps(charts_data, indent=2)};
const UPDATED = "{ts}";

type ChartEntry = {{
  label: string; ylabel: string;
  cats: string[]; mean: (number|null)[]; upper: (number|null)[]; lower: (number|null)[];
}};

function ShadedChart({{ d, accentColor }}: {{ d: ChartEntry; accentColor: string }}) {{
  const theme = useHostTheme();
  const W = 520, H = 190;
  const padL = 52, padR = 12, padT = 12, padB = 28;
  const cW = W - padL - padR, cH = H - padT - padB;

  const mean  = d.mean  as number[];
  const upper = d.upper as number[];
  const lower = d.lower as number[];
  const cats  = d.cats;
  const n     = cats.length;

  const allVals = [...mean, ...upper, ...lower].filter(v => v != null) as number[];
  const rawMax = allVals.length ? Math.max(...allVals) : 1;
  const rawMin = allVals.length ? Math.min(0, Math.min(...allVals)) : 0;
  const pad = (rawMax - rawMin) * 0.08 || 0.5;
  const minY = rawMin, maxY = rawMax + pad;
  const rng = maxY - minY || 1;

  const xS = (i: number) => padL + (n > 1 ? (i / (n - 1)) : 0.5) * cW;
  const yS = (v: number) => padT + cH - ((v - minY) / rng) * cH;

  // shaded polygon: upper forward, lower backward
  const uPts = upper.map((v, i) => `${{xS(i)}},${{yS(v ?? mean[i] ?? minY)}}`).join(" ");
  const lPts = [...lower].reverse().map((v, i) => `${{xS(n-1-i)}},${{yS(v ?? mean[n-1-i] ?? minY)}}`).join(" ");
  const shadeD = `M ${{uPts}} L ${{lPts}} Z`;

  // mean polyline
  const lineD = mean.map((v, i) => `${{i===0?"M":"L"}}${{xS(i)}},${{yS(v ?? minY)}}`).join(" ");

  // y ticks
  const nTicks = 4;
  const yTicks = Array.from({{length: nTicks+1}}, (_,i) => {{
    const val = minY + (i/nTicks)*rng;
    return {{ val, y: yS(val) }};
  }});

  // x ticks: ~6 labels
  const xStep = Math.max(1, Math.floor(n / 6));
  const xTickIdxs = Array.from({{length: n}}, (_,i) => i)
    .filter(i => i % xStep === 0 || i === n-1);

  const fmt = (v: number) => v >= 1000 ? `${{(v/1000).toFixed(1)}}k` : v.toFixed(v < 1 ? 2 : 0);

  return (
    <svg width={{W}} height={{H}} style={{{{ overflow:"visible", display:"block" }}}}>
      {{/* grid */}}
      {{yTicks.map((t,i) => (
        <g key={{i}}>
          <line x1={{padL}} y1={{t.y}} x2={{W-padR}} y2={{t.y}}
            stroke={{theme.stroke.secondary}} strokeWidth={{0.5}} />
          <text x={{padL-5}} y={{t.y+4}} textAnchor="end" fontSize={{10}}
            fill={{theme.text.tertiary}}>{{fmt(t.val)}}</text>
        </g>
      ))}}
      {{/* y-axis label */}}
      <text x={{11}} y={{padT + cH/2}} textAnchor="middle" fontSize={{11}}
        fill={{theme.text.secondary}}
        style={{{{ transform:`rotate(-90deg)`, transformOrigin:`11px ${{padT+cH/2}}px` }}}}>
        {{d.ylabel}}
      </text>
      {{/* shade */}}
      <path d={{shadeD}} fill={{accentColor}} fillOpacity={{0.18}} />
      {{/* mean line */}}
      <path d={{lineD}} fill="none" stroke={{accentColor}} strokeWidth={{2.2}}
        strokeLinecap="round" strokeLinejoin="round" />
      {{/* x ticks */}}
      {{xTickIdxs.map(i => (
        <text key={{i}} x={{xS(i)}} y={{H-6}} textAnchor="middle" fontSize={{10}}
          fill={{theme.text.tertiary}}>{{cats[i]}}</text>
      ))}}
      {{/* axes */}}
      <line x1={{padL}} y1={{padT}} x2={{padL}} y2={{padT+cH}}
        stroke={{theme.stroke.primary}} strokeWidth={{1}} />
      <line x1={{padL}} y1={{padT+cH}} x2={{W-padR}} y2={{padT+cH}}
        stroke={{theme.stroke.primary}} strokeWidth={{1}} />
    </svg>
  );
}}

const ACCENT_COLORS = ["#4C8EF5","#E87040","#49B885","#A070D8"];

export default function TrainingMonitor() {{
  const theme = useHostTheme();
  const envKeys = Object.keys(CHARTS);
  return (
    <Stack gap={{20}} style={{{{ padding:20 }}}}>
      <Row align="center" justify="space-between">
        <H1>PPO Training Monitor</H1>
        <Text style={{{{ color:theme.text.tertiary, fontSize:12 }}}}>
          Mean ± 1 std across seeds · Updated {{UPDATED}}
        </Text>
      </Row>
      <Grid columns={{2}} gap={{16}}>
        {{envKeys.map((env, ei) => {{
          const d = (CHARTS as any)[env] as ChartEntry;
          return (
            <Card key={{env}}>
              <CardHeader><H2>{{d.label}}</H2></CardHeader>
              <CardBody style={{{{ paddingTop:4 }}}}>
                <ShadedChart d={{d}} accentColor={{ACCENT_COLORS[ei % ACCENT_COLORS.length]}} />
                <Text style={{{{ color:theme.text.tertiary, fontSize:11, marginTop:4 }}}}>
                  Learner iterations
                </Text>
              </CardBody>
            </Card>
          );
        }})}}
      </Grid>
    </Stack>
  );
}}
""")

os.makedirs(os.path.dirname(CANVAS_PATH), exist_ok=True)
with open(CANVAS_PATH, "w") as f:
    f.write(CANVAS)

print(f"[{ts}] Canvas written → {CANVAS_PATH}")
print(f"  Envs with data: {list(charts_data.keys())}")
for env, d in charts_data.items():
    print(f"  {env}: cats={len(d['cats'])}, mean_pts={sum(1 for v in d['mean'] if v is not None)}")
