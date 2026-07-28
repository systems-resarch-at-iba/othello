interface CellProps {
  value: -1 | 0 | 1
  isLegalMove: boolean
  /** "Show move hints": a plain legality marker, independent of the AI. */
  showMoveDot: boolean
  /** "Show AI hints": the AI's suggested probability for this square, 0 when off or not applicable. Drives the yellow highlight intensity. */
  hintProb: number
  onClick: () => void
}

export function Cell({ value, isLegalMove, showMoveDot, hintProb, onClick }: CellProps) {
  const highlighted = hintProb > 0

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={!isLegalMove}
      // `!isLegalMove` is a plain boolean negation, so this attribute can
      // never actually differ between server and client renders -- but some
      // browser extensions (accessibility/testing tools that force-enable
      // disabled controls) rewrite the `disabled` DOM property before React
      // hydrates, which React then reports as a mismatch it can't resolve.
      // That's exactly the case suppressHydrationWarning exists for: a
      // specific attribute known to be altered by something outside React's
      // control, not application logic to fix.
      suppressHydrationWarning
      aria-label={value === 1 ? 'White piece' : value === -1 ? 'Black piece' : isLegalMove ? 'Legal move' : 'Empty'}
      style={{
        width: '100%',
        aspectRatio: '1',
        background: highlighted
          ? `rgba(234, 179, 8, ${Math.min(0.2 + hintProb * 0.7, 0.9)})`
          : '#0d9488',
        border: '1px solid rgba(0,0,0,0.15)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        cursor: isLegalMove ? 'pointer' : 'default',
        padding: 0,
        transition: 'background 150ms ease',
      }}
    >
      {value !== 0 && (
        <span
          style={{
            width: '80%',
            aspectRatio: '1',
            borderRadius: '50%',
            background: value === 1 ? '#f5f5f4' : '#15161b',
          }}
        />
      )}
      {value === 0 && isLegalMove && showMoveDot && !highlighted && (
        <span
          style={{
            width: '25%',
            aspectRatio: '1',
            borderRadius: '50%',
            background: 'rgba(255,255,255,0.4)',
          }}
        />
      )}
    </button>
  )
}
