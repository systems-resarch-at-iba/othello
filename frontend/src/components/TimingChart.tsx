'use client'

import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import {
  CHART_HEIGHT,
  chartAxisStyle,
  chartGridStroke,
  chartLineStroke,
  tooltipContentStyle,
  tooltipItemStyle,
  tooltipLabelStyle,
} from '../lib/chart-theme'

interface TimingChartProps {
  /** Response times in ms, one per AI move, oldest first. */
  history: number[]
}

/**
 * Fixed-height regardless of how much data there is (including none), so
 * this never causes the layout shift the single inline "AI responded in
 * Xms" line used to: that line's own appearance/disappearance moved
 * everything below it up and down every time the AI moved.
 */
export function TimingChart({ history }: TimingChartProps) {
  const data = history.map((ms, i) => ({ move: i + 1, ms: Math.round(ms) }))

  return (
    <div style={{ width: '100%', height: CHART_HEIGHT }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 12, left: 4, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={chartGridStroke} vertical={false} />
          <XAxis
            dataKey="move"
            {...chartAxisStyle}
            label={{ value: 'Move', position: 'insideBottom', offset: -4, ...chartAxisStyle.tick }}
          />
          {/* No axis label here on purpose: at this width, an "ms" label
              rotated inside the axis collides with the tick numbers. Units
              are already stated in the tooltip below instead. */}
          <YAxis {...chartAxisStyle} width={44} />
          <Tooltip
            cursor={{ stroke: chartLineStroke, strokeWidth: 1, strokeDasharray: '3 3' }}
            formatter={(value) => [`${value}ms`, 'Response time']}
            labelFormatter={(label) => `Move ${label}`}
            contentStyle={tooltipContentStyle}
            labelStyle={tooltipLabelStyle}
            itemStyle={tooltipItemStyle}
          />
          <Line
            type="monotone"
            dataKey="ms"
            stroke={chartLineStroke}
            strokeWidth={2}
            dot={{ r: 3, fill: chartLineStroke }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
