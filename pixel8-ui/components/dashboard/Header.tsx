// components/dashboard/Header.tsx
import Image from "next/image";
import { Montserrat } from "next/font/google";

const mont = Montserrat({
  subsets: ["latin"],
  weight: ["600"], // pick what you use
});

const montThin = Montserrat({
  subsets: ["latin"],
  weight: ["500"], // pick what you use
});

export default function Header() {
  return (
    <header className="relative overflow-hidden bg-white">
      <div className="flex flex-col gap-4 p-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          {/* Logos + Signals aligned, subtitle under mark */}
          <div className="grid grid-cols-[auto_auto_auto] grid-rows-[auto_auto] items-center gap-x-2">
            {/* Row 1: mashreq logo, mashreq mark, signals (all aligned) */}
            <Image
              src="/mashreq-logo.svg"
              alt="Mashreq logo"
              width={130}
              height={50}
              priority
              className="row-start-1 col-start-1"
            />

            <Image
              src="/mashreq-mark.svg"
              alt="Mashreq mark"
              width={130}
              height={130}
              priority
              className="row-start-1 col-start-2"
            />

            <h1
              className={`${mont.className} row-start-1 col-start-3 text-3xl font-semibold tracking-wide`}
            >
              • SIGNALS
            </h1>

            {/* Row 2: subtitle starts exactly where the mark starts */}
            <p
              className={`${montThin.className} row-start-2 col-start-2 col-span-2 -mt-4 text-sm text-[#ee912e]`}
            >
              Public chatter → risk prioritisation
            </p>
          </div>
        </div>

        {/* Right side status */}
        <div className="flex items-center gap-1">
          <div className={`${montThin.className} text-sm text-[#6a6a6a]`}>Signal Monitoring</div>
          <div className={`${mont.className} text-sm font-bold text-[#6a6a6a]`}>Dashboard</div>
        </div>
      </div>

      <div className="h-1.5 w-full bg-[linear-gradient(90deg,#e0462a_0%,#f7bb34_100%)]" />
    </header>
  );
}
