import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

// Initialize the Geist font with Latin subset
const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

// Initialize the Geist Mono font with Latin subset
const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

// Define metadata for better SEO
export const metadata: Metadata = {
  title: "Next.js Beginner Template",
  description: "A beginner-friendly Next.js template with TailwindCSS and TypeScript",
  keywords: ["Next.js", "React", "TailwindCSS", "TypeScript", "Template"],
  authors: [{ name: "Created with Cursor Agent" }],
  creator: "Cursor Agent",
  publisher: "Cursor Agent",
  openGraph: {
    title: "Next.js Beginner Template",
    description: "A beginner-friendly Next.js template with TailwindCSS and TypeScript",
    url: "https://nextjs.org/",
    siteName: "Next.js Beginner Template",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "Next.js Beginner Template",
      },
    ],
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Next.js Beginner Template",
    description: "A beginner-friendly Next.js template with TailwindCSS and TypeScript",
    images: ["/og-image.png"],
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100`}
      >
        {children}
      </body>
    </html>
  );
}
