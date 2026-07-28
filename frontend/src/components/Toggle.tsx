import type { ReactNode } from 'react'

interface ToggleProps {
  checked: boolean
  onChange: (checked: boolean) => void
  children: ReactNode
}

/** A real switch, not a bare `<input type="checkbox">`: the native checkbox stays for a11y/keyboard support but is visually hidden, styled by the two spans that track its `:checked` state via the `peer` variant. */
export function Toggle({ checked, onChange, children }: ToggleProps) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-3 font-sans text-sm text-ink">
      <span>{children}</span>
      <span className="relative inline-flex h-5 w-9 shrink-0 items-center">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="peer sr-only"
        />
        <span className="absolute inset-0 rounded-full bg-hairline-strong transition-colors peer-checked:bg-signal" />
        <span className="absolute left-0.5 h-4 w-4 rounded-full bg-white shadow-sm transition-transform peer-checked:translate-x-4" />
      </span>
    </label>
  )
}
