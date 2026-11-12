import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import Script from "next/script";
import { Providers } from "@/components/Providers";
import { Shell } from "@/components/Shell";
import { themeInitScript } from "@/components/theme";
import "./globals.css";

const body = IBM_Plex_Sans({ variable: "--font-body", subsets: ["latin"], weight: ["400", "500", "600", "700"] });
const code = IBM_Plex_Mono({ variable: "--font-code", subsets: ["latin"], weight: ["400", "500"] });

export const metadata: Metadata = { title: { default: "TuneLab", template: "%s · TuneLab" }, description: "Genuine LoRA fine-tuning and evaluation on Apple Silicon." };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" data-theme="light" suppressHydrationWarning className={`${body.variable} ${code.variable}`}>
      <body className="min-h-dvh">
        <Script id="theme" strategy="beforeInteractive">{themeInitScript}</Script>
        <Providers>
          <Shell>{children}</Shell>
        </Providers>
      </body>
    </html>
  );
}
