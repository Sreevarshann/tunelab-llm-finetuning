"use client";

import dynamic from "next/dynamic";
import type { Data, Layout } from "plotly.js";
import { useTheme } from "./theme";
import { Skeleton } from "./ui";

const Plotly = dynamic(
  async () => {
    const [{ default: factory }, plotly] = await Promise.all([import("react-plotly.js/factory"), import("plotly.js-dist-min")]);
    return factory(plotly.default ?? plotly);
  },
  { ssr: false, loading: () => <Skeleton className="h-64 w-full" /> },
);

export const COLORS = { light: { base: "#6650d8", tuned: "#0b8a68", warn: "#b06d10", bad: "#b5283c", muted: "#8b8d93" }, dark: { base: "#a596ff", tuned: "#45d1a5", warn: "#f0b45a", bad: "#ff8b9b", muted: "#6b6e75" } };
export function usePalette() {
  const { theme } = useTheme();
  return COLORS[theme === "dark" ? "dark" : "light"];
}

export function Plot({ data, layout, height = 280, label }: { data: Data[]; layout?: Partial<Layout>; height?: number; label: string }) {
  const { theme } = useTheme();
  const ink = theme === "dark" ? "#e8e8e6" : "#18191c";
  const grid = theme === "dark" ? "#2c2e33" : "#e6e6e1";
  return (
    <div role="img" aria-label={label}>
      <Plotly
        data={data}
        useResizeHandler
        style={{ width: "100%", height }}
        config={{ displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] }}
        layout={{ autosize: true, height, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: ink, size: 12, family: "IBM Plex Sans, system-ui, sans-serif" }, margin: { l: 48, r: 12, t: 8, b: 40 }, legend: { orientation: "h", y: -0.22 }, xaxis: { gridcolor: grid, zeroline: false }, yaxis: { gridcolor: grid, zeroline: false }, ...layout }}
      />
    </div>
  );
}
