import { NextResponse } from "next/server";
import path from "path";
import { promises as fs } from "fs";

export async function GET() {
  try {
    // pixel8-ui/app/api/dashboard/route.ts
    // go up: pixel8-ui -> pixel8
    const filePath = path.join(process.cwd(), "..", "data", "dashboard_summary.json");
    const raw = await fs.readFile(filePath, "utf8");
    const json = JSON.parse(raw);

    return NextResponse.json(json, { status: 200 });
  } catch (err: any) {
    return NextResponse.json(
      { error: "Failed to read dashboard_summary.json", details: err?.message ?? String(err) },
      { status: 500 }
    );
  }
}
