import { useEffect, useRef, useState } from "react";

/**
 * Custom "audit stamp" cursor.
 * - A tight dot tracks the pointer exactly.
 * - A trailing ring eases toward it (lerp), giving weight to movement.
 * - Hovering [data-cursor="stamp"] elements morphs the ring into a
 *   rotated stamp bracket; clicking plays a quick stamp-down pulse.
 * Disabled entirely on touch devices and prefers-reduced-motion.
 */
export default function Cursor() {
  const dotRef = useRef(null);
  const ringRef = useRef(null);
  const [enabled, setEnabled] = useState(false);
  const [label, setLabel] = useState("");
  const state = useRef({ mx: 0, my: 0, rx: 0, ry: 0, hover: false, down: false });

  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const touch = window.matchMedia("(pointer: coarse)").matches;
    if (reduce || touch) return;

    setEnabled(true);
    document.documentElement.classList.add("custom-cursor");

    const onMove = (e) => {
      state.current.mx = e.clientX;
      state.current.my = e.clientY;
      const target = e.target.closest("[data-cursor]");
      state.current.hover = !!target;
      setLabel(target?.dataset.cursor === "stamp" ? "AUDIT" : "");
    };
    const onDown = () => (state.current.down = true);
    const onUp = () => (state.current.down = false);

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mousedown", onDown);
    window.addEventListener("mouseup", onUp);

    let raf;
    const tick = () => {
      const s = state.current;
      s.rx += (s.mx - s.rx) * 0.18;
      s.ry += (s.my - s.ry) * 0.18;

      if (dotRef.current) {
        dotRef.current.style.transform = `translate(${s.mx - 3}px, ${s.my - 3}px)`;
      }
      if (ringRef.current) {
        const scale = s.hover ? (s.down ? 0.85 : 1.15) : s.down ? 0.8 : 1;
        const rot = s.hover ? -8 : 0;
        ringRef.current.style.transform = `translate(${s.rx - 18}px, ${s.ry - 18}px) rotate(${rot}deg) scale(${scale})`;
        ringRef.current.dataset.hover = s.hover ? "1" : "0";
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mousedown", onDown);
      window.removeEventListener("mouseup", onUp);
      cancelAnimationFrame(raf);
      document.documentElement.classList.remove("custom-cursor");
    };
  }, []);

  if (!enabled) return null;

  return (
    <div className="pointer-events-none fixed inset-0 z-[999]">
      <div
        ref={dotRef}
        className="absolute top-0 left-0 h-[6px] w-[6px] rounded-full bg-stamp"
        style={{ willChange: "transform" }}
      />
      <div
        ref={ringRef}
        className="absolute top-0 left-0 h-9 w-9 rounded-md border-[1.5px] border-stamp/70 transition-[border-radius] duration-150 flex items-center justify-center"
        style={{ willChange: "transform" }}
      >
        <span
          className="font-mono text-[8px] tracking-[0.2em] text-stamp/0 data-[on=1]:text-stamp/90 transition-colors"
          data-on={label ? "1" : "0"}
        >
          {label}
        </span>
      </div>
    </div>
  );
}
