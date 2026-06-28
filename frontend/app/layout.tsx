import type { Metadata } from "next";
import { Fraunces, Hanken_Grotesk, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";
import { Backdrop } from "@/components/Backdrop";
import { Nav } from "@/components/Nav";

const display = Fraunces({
  subsets: ["latin"], weight: ["400", "600", "900"], style: ["normal", "italic"],
  variable: "--font-display", display: "swap",
});
const mono = IBM_Plex_Mono({
  subsets: ["latin"], weight: ["400", "500", "600"], variable: "--font-mono", display: "swap",
});
const body = Hanken_Grotesk({
  subsets: ["latin"], weight: ["400", "500", "700"], variable: "--font-body", display: "swap",
});

export const metadata: Metadata = {
  title: "ReefScan — coral health analysis",
  description: "Segment, classify, and quantify coral reef health with conformal uncertainty.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${mono.variable} ${body.variable}`}>
      <body>
        <Backdrop />
        <div style={{ position: "relative", zIndex: 1 }}>
          <Nav />
          <main className="mx-auto w-full max-w-[1180px] px-5 pb-24 pt-7 md:px-8">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
