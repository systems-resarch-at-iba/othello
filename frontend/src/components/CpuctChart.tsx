'use client'

import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { CpuctType } from '../lib/types'
import {
  CHART_HEIGHT,
  chartAxisStyle,
  chartGridStroke,
  chartLineStroke,
  tooltipContentStyle,
  tooltipItemStyle,
  tooltipLabelStyle,
} from '../lib/chart-theme'

interface CpuctChartProps {
  cpuctType: Exclude<CpuctType, 'static'>
  cPuct: number
}

// Mirrors MCTS.exp_scale exactly (engine/mcts.py): approach + (start -
// approach) * exp(-k * visits), k's default of 0.001 unchanged by any
// caller. 'increment' starts at 0.5 and rises toward cPuct as visits grow;
// 'decrement' starts at cPuct and decays toward 0.5.
const DECAY_K = 0.001
const MAX_VISITS = 2000 // matches the MCTS simulations field's own cap

function expScale(visits: number, start: number, approach: number): number {
  return approach + (start - approach) * Math.exp(-DECAY_K * visits)
}

const POINTS = 60

export function CpuctChart({ cpuctType, cPuct }: CpuctChartProps) {
  const [start, approach] = cpuctType === 'increment' ? [0.5, cPuct] : [cPuct, 0.5]
  const data = Array.from({ length: POINTS + 1 }, (_, i) => {
    const visits = Math.round((i / POINTS) * MAX_VISITS)
    return { visits, cpuct: Number(expScale(visits, start, approach).toFixed(3)) }
  })

  return (
    <div style={{ width: '100%', height: CHART_HEIGHT }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 12, left: 4, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={chartGridStroke} vertical={false} />
          <XAxis
            dataKey="visits"
            {...chartAxisStyle}
            label={{ value: 'Visits', position: 'insideBottom', offset: -4, ...chartAxisStyle.tick }}
          />
          <YAxis {...chartAxisStyle} width={44} domain={[0, 'dataMax']} />
          <Tooltip
            cursor={{ stroke: chartLineStroke, strokeWidth: 1, strokeDasharray: '3 3' }}
            formatter={(value) => [value, 'c_puct']}
            labelFormatter={(label) => `${label} visits`}
            contentStyle={tooltipContentStyle}
            labelStyle={tooltipLabelStyle}
            itemStyle={tooltipItemStyle}
          />
          <Line type="monotone" dataKey="cpuct" stroke={chartLineStroke} strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
