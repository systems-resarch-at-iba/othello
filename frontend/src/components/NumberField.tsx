import { useEffect, useState } from 'react'

interface NumberFieldProps {
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step?: number
}

function clamp(n: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, n))
}

/**
 * A number input backed by its own string state instead of formatting
 * `value` directly into the DOM: a plain controlled `<input value={n}>`
 * fights the user mid-edit (clearing the field round-trips through
 * `Number('') === 0`, so React re-renders "0" while they're still typing,
 * and the next digit lands after it as "01" instead of replacing it).
 * The number only gets parsed, clamped, and pushed to `onChange` once
 * there's a real value to parse; out-of-range or invalid text is only
 * corrected on blur, not on every keystroke.
 */
export function NumberField({ value, onChange, min, max, step = 1 }: NumberFieldProps) {
  const [text, setText] = useState(String(value))

  // Keeps the field in sync if `value` changes from elsewhere (e.g. a
  // parent reset); harmless no-op while this field itself is the source
  // of the change, since it's setting the same string it already has.
  useEffect(() => {
    setText(String(value))
  }, [value])

  return (
    <input
      type="number"
      min={min}
      max={max}
      step={step}
      value={text}
      onChange={(e) => {
        const raw = e.target.value
        setText(raw)
        const parsed = Number(raw)
        // Only propagate values already in range while typing: clamping
        // an in-progress out-of-range number (e.g. "0" on the way to
        // typing "0.5") would push a corrected value back down through
        // `value`, which the effect above then re-syncs into `text`,
        // overwriting what the user just typed before they can finish it.
        if (raw.trim() !== '' && !Number.isNaN(parsed) && parsed >= min && parsed <= max) {
          onChange(parsed)
        }
      }}
      onBlur={() => {
        const parsed = Number(text)
        const next = Number.isNaN(parsed) ? min : clamp(parsed, min, max)
        setText(String(next))
        onChange(next)
      }}
      className="w-full rounded-sm border border-hairline-strong bg-paper-raised px-2 py-1.5 font-sans text-sm text-ink transition-colors focus:border-signal focus:outline-none focus:ring-2 focus:ring-signal/30"
    />
  )
}
