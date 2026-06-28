// Soft caustic-light backdrop — three blurred radial tints drifting like underwater light.
export function Backdrop() {
  return (
    <div aria-hidden style={{ position: "fixed", inset: 0, zIndex: 0, pointerEvents: "none", overflow: "hidden" }}>
      <div style={{
        position: "absolute", top: "-25%", left: "-15%", width: "70%", height: "80%",
        background: "radial-gradient(circle at 50% 50%, var(--tint-blue) 0%, transparent 62%)",
        filter: "blur(20px)", opacity: 0.7, animation: "rsCaustic1 22s ease-in-out infinite",
      }} />
      <div style={{
        position: "absolute", bottom: "-30%", right: "-20%", width: "75%", height: "85%",
        background: "radial-gradient(circle at 50% 50%, var(--tint-sage) 0%, transparent 60%)",
        filter: "blur(22px)", opacity: 0.55, animation: "rsCaustic2 28s ease-in-out infinite",
      }} />
      <div style={{
        position: "absolute", top: "30%", right: "8%", width: "40%", height: "50%",
        background: "radial-gradient(circle at 50% 50%, var(--tint-sand) 0%, transparent 65%)",
        filter: "blur(26px)", opacity: 0.5, animation: "rsCaustic1 34s ease-in-out infinite reverse",
      }} />
    </div>
  );
}
