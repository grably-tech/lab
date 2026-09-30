// Static server for the lab. Supports Range requests: Chrome needs them to seek inside mp4 clips.
// POST /api/live-session stores a live calibration session (landmarks + result, no images) in data/live-sessions/ for
// offline debugging with the same code.
import { mkdir } from "node:fs/promises";
import { join, normalize } from "node:path";

const root = normalize(join(import.meta.dir, ".."));
const port = Number(process.env.PORT ?? 5180);
const types: Record<string, string> = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".json": "application/json", ".mp4": "video/mp4" };

Bun.serve({
  port,
  async fetch(req) {
    if (req.method === "POST" && new URL(req.url).pathname === "/api/live-session") {
      const dir = join(root, "data", "live-sessions");
      await mkdir(dir, { recursive: true });
      const name = `${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
      await Bun.write(join(dir, name), await req.text());
      console.log(`live session saved: data/live-sessions/${name}`);
      return Response.json({ saved: `data/live-sessions/${name}` });
    }
    const path = normalize(join(root, decodeURIComponent(new URL(req.url).pathname)));
    if (!path.startsWith(root)) return new Response("forbidden", { status: 403 });
    const file = Bun.file(path.endsWith("/") ? join(path, "index.html") : path);
    if (!(await file.exists())) return new Response("not found", { status: 404 });
    const headers = { "content-type": types[path.slice(path.lastIndexOf("."))] ?? file.type, "accept-ranges": "bytes", "cache-control": "no-store" };
    const range = req.headers.get("range")?.match(/^bytes=(\d*)-(\d*)$/);
    if (!range) return new Response(file, { headers });
    const start = range[1] ? Number(range[1]) : file.size - Number(range[2]);
    const end = range[1] && range[2] ? Math.min(Number(range[2]), file.size - 1) : file.size - 1;
    return new Response(file.slice(start, end + 1), {
      status: 206,
      headers: { ...headers, "content-range": `bytes ${start}-${end}/${file.size}`, "content-length": String(end - start + 1) },
    });
  },
});
console.log(`stereo-lab on http://localhost:${port}/`);
