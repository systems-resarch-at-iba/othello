'use client'

import { useEffect, useState } from 'react'
import { checkHealth } from '../lib/api-client'
import type { ModelHealth } from '../lib/api-client'

interface ModelCardProps {
  apiBaseUrl: string
}

function formatParameterCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return String(n)
}

/**
 * Purely informational, not configurable -- this is "about the model," the
 * counterpart to Settings rather than part of it, which is also why it sits
 * in its own panel on the opposite side of the board rather than sharing
 * Settings' column. Version/architecture/parameter count come from /health
 * at request time so they never drift from what's actually loaded; the
 * distillation story itself is fixed project history, not something the
 * backend needs to serve.
 */
export function ModelCard({ apiBaseUrl }: ModelCardProps) {
  const [health, setHealth] = useState<ModelHealth | null>(null)

  useEffect(() => {
    let cancelled = false
    checkHealth(apiBaseUrl)
      .then((result) => {
        if (!cancelled) setHealth(result)
      })
      .catch(() => {
        // A failed health check just means the card stays empty; an actual
        // /move failure already surfaces its own error in the game itself.
      })
    return () => {
      cancelled = true
    }
  }, [apiBaseUrl])

  if (!health) return null

  return (
    <div className="font-sans text-sm text-ink">
      <h3 className="mb-3 border-b border-hairline pb-3 text-xs font-semibold uppercase tracking-widest text-ink-muted">
        About this model
      </h3>
      <dl className="flex flex-col gap-2">
        <div className="flex justify-between gap-2">
          <dt className="text-ink-muted">Architecture</dt>
          <dd className="font-medium">{health.architecture}</dd>
        </div>
        <div className="flex justify-between gap-2">
          <dt className="text-ink-muted">Version</dt>
          <dd className="font-medium">{health.modelVersion}</dd>
        </div>
        <div className="flex justify-between gap-2">
          <dt className="text-ink-muted">Parameters</dt>
          <dd className="font-medium">{formatParameterCount(health.parameterCount)}</dd>
        </div>
      </dl>
      <p className="mt-3 border-t border-hairline pt-3 font-serif text-sm leading-relaxed text-ink-muted">
        Distilled from a larger teacher network (~18.7M parameters) via a KL-divergence loss
        on the teacher&apos;s own move probabilities and value estimates, rather than trained
        from self-play directly.
      </p>
    </div>
  )
}
