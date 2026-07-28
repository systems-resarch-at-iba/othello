import { useEffect, useRef, useState } from 'react'

interface DropdownOption<T extends string | number> {
  value: T
  label: string
}

interface DropdownProps<T extends string | number> {
  value: T
  options: DropdownOption<T>[]
  onChange: (value: T) => void
}

/**
 * A button-plus-panel dropdown, the same pattern the site's own theme
 * switcher uses (components/layout/theme-switcher.tsx), rather than a
 * native `<select>`: only this level of control gets the option list to
 * actually pick up the site's colors, since a native `<select>`'s open
 * popup is OS-rendered and can't be restyled.
 */
export function Dropdown<T extends string | number>({ value, options, onChange }: DropdownProps<T>) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    function onPointerDown(e: PointerEvent) {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('pointerdown', onPointerDown)
    return () => document.removeEventListener('pointerdown', onPointerDown)
  }, [open])

  const current = options.find((o) => o.value === value)

  return (
    <div ref={rootRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full cursor-pointer items-center justify-between gap-2 rounded-sm border border-hairline-strong bg-paper-raised px-2 py-1.5 text-left font-sans text-sm text-ink transition-colors hover:border-hairline-strong focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-signal"
      >
        <span className="truncate">{current?.label}</span>
        <svg
          aria-hidden="true"
          viewBox="0 0 12 8"
          className="h-2 w-2.5 shrink-0 text-signal"
        >
          <path
            d="M1 1l5 5 5-5"
            stroke="currentColor"
            strokeWidth="1.5"
            fill="none"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>

      {open && (
        // w-max + min-w-full: never narrower than the trigger button, but
        // grows to fit the longest option label instead of wrapping it.
        // Anchored to the right edge (not left) so that growth extends back
        // over the panel rather than off the right side of the viewport,
        // since this sits inside a right-hand sidebar.
        <div className="absolute right-0 z-10 mt-1 w-max min-w-full rounded-md border border-hairline bg-paper-raised p-1 shadow-nav">
          {options.map((option) => (
            <button
              key={option.value}
              type="button"
              onClick={() => {
                onChange(option.value)
                setOpen(false)
              }}
              className={`block w-full cursor-pointer whitespace-nowrap rounded-sm px-2 py-1.5 text-left font-sans text-sm transition-colors ${
                option.value === value ? 'bg-signal-dim text-signal-ink' : 'text-ink hover:bg-paper'
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
