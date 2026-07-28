/**
 * Shared chart styling for TimingChart/CpuctChart, as CSS custom property
 * references rather than literal hex values: the site's theme switcher
 * overrides these `--color-*` variables at runtime (see globals.css), and
 * since these are plain strings passed into SVG `stroke`/`fill`/`color`
 * props, the browser resolves them the same way it would for any other
 * themed element -- a chart styled with literal hex would otherwise stay
 * fixed to one theme's colors (this is what caused the light-on-light /
 * white-on-white tooltip in dark mode before).
 */
export const CHART_HEIGHT = 220

export const chartAxisStyle = {
  tick: { fontSize: 12, fill: 'var(--color-ink-muted)' },
  axisLine: { stroke: 'var(--color-hairline)' },
  tickLine: false as const,
}

export const chartGridStroke = 'var(--color-hairline)'
export const chartLineStroke = 'var(--color-signal)'

export const tooltipContentStyle = {
  fontSize: 13,
  borderRadius: 6,
  backgroundColor: 'var(--color-paper-raised)',
  borderColor: 'var(--color-hairline)',
  color: 'var(--color-ink)',
}

export const tooltipLabelStyle = { color: 'var(--color-ink)' }
export const tooltipItemStyle = { color: 'var(--color-ink)' }
