import type { Metadata } from "next";
import { JetBrains_Mono, Manrope, Newsreader } from "next/font/google";
import "./globals.css";
import { Backdrop } from "@/components/Backdrop";
import { Nav } from "@/components/Nav";

const display = Newsreader({
  subsets: ["latin"], weight: ["400", "500"], style: ["normal", "italic"],
  variable: "--font-display", display: "swap",
});
const body = Manrope({
  subsets: ["latin"], weight: ["400", "500", "600", "700", "800"],
  variable: "--font-body", display: "swap",
});
const mono = JetBrains_Mono({
  subsets: ["latin"], weight: ["400", "500", "600"], variable: "--font-mono", display: "swap",
});

export const metadata: Metadata = {
  title: "ReefScan — coral health analysis",
  description: "Segment, classify, and quantify coral reef health with conformal uncertainty.",
};

// set the theme before paint to avoid a flash (light default)
const themeInit = `(function(){try{var t=localStorage.getItem('reefscan-theme')||'light';document.documentElement.setAttribute('data-theme',t);}catch(e){document.documentElement.setAttribute('data-theme','light');}})();`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" data-theme="light" className={`${display.variable} ${mono.variable} ${body.variable}`}>
      <head><script dangerouslySetInnerHTML={{ __html: themeInit }} /></head>
      <body>
        <Backdrop />
        <div style={{ position: "relative", zIndex: 1 }}>
          <Nav />
          <main className="mx-auto w-full max-w-[1240px] px-4 pb-24 pt-2 md:px-8">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
