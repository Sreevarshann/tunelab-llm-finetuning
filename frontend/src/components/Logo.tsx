export function Logo({ className = "size-7" }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden>
      <rect width="32" height="32" rx="8" fill="var(--ink)" />
      <path d="M7 21c3-9 6-9 9 0s6 9 9 0" fill="none" stroke="var(--tuned)" strokeWidth="2.4" strokeLinecap="round" />
      <path d="M7 13h18" stroke="var(--base)" strokeWidth="2" strokeLinecap="round" strokeDasharray="2 3" />
    </svg>
  );
}
