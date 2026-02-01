// app/page.tsx
import Header from "@/components/dashboard/Header";
import DashboardClient from "@/components/dashboard/DashboardClient";

export default function Page() {
  return (
    <main className="min-h-screen bg-[linear-gradient(to_bottom,#f4f2f0_0%,#f4f2f0_70%,#ffffff_100%)] text-[#f9753b]">
      <div className="mx-auto max-w-7xl px-0 py-0">
        {/* Mashreq-style header (logo + orange strip) */}
        <Header />

        <div className="mt-6">
          <DashboardClient />
        </div>

        <footer className="mt-10 border-t border-zinc-800 pt-6 text-xs text-zinc-400">
          Mashreq Signals Command Centre • Synthetic Data • Powered by Pixel8
        </footer>
      </div>
    </main>
  );
}
