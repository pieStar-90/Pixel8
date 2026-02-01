// components/dashboard/KpiCard.tsx
import { Montserrat } from "next/font/google";

type Tone = "danger" | "warning" | "neutral" | "accent";

const mont = Montserrat({
  subsets: ["latin"],
  weight: ["500"], // pick what you use
});

export default function KpiCard({
  title,
  value,
  tone = "neutral",
  onClick,
}: {
  title: string;
  value: number;
  tone?: Tone;
  onClick?: () => void;
}) {
  const ring =
    tone === "danger"
      ? "ring-red-500/20"
      : tone === "warning"
      ? "ring-amber-500/20"
      : tone === "accent"
      ? "ring-orange-500/20"
      : "ring-zinc-400/40";

  const chip =
    tone === "danger"
      ? "bg-red-500/15 text-red-300"
      : tone === "warning"
      ? "bg-amber-500/15 text-amber-400"
      : tone === "accent"
      ? "bg-orange-500/15 text-orange-400"
      : "bg-zinc-800/60 text-zinc-200";

  const isClickable = typeof onClick === "function";

  return (
    <div
      role={isClickable ? "button" : undefined}
      tabIndex={isClickable ? 0 : undefined}
      onClick={onClick}
      onKeyDown={(e) => {
        if (!isClickable) return;
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onClick?.();
        }
      }}
      className={[
        "rounded-2xl bg-white p-4 ring-2 transition-shadow duration-200 hover:shadow-lg",
        ring,
        isClickable ? "cursor-pointer select-none focus:outline-none focus:ring-2 focus:ring-orange-400/40" : "",
      ].join(" ")}
    >
      <div className="flex items-center justify-between">
        <div className={`${mont.className} text-ml text-[#6b6b6b]`}>{title}</div>
        <div className={`rounded-full px-2 py-1 text-xs ${chip}`}>Live</div>
      </div>
      <div className="mt-3 text-3xl font-semibold tracking-tight">{value}</div>
    </div>
  );
}
