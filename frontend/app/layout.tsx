import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Hashed Assistant — test harness",
  description: "Exercises the sales assistant API end to end.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
