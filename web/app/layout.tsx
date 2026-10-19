import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Music Atlas",
  description: "Explore how music becomes data.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return <html lang="pl"><body>{children}</body></html>;
}
