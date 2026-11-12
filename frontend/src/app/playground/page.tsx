import type { Metadata } from "next";
import { Suspense } from "react";
import { Playground } from "@/components/Playground";

export const metadata: Metadata = { title: "Playground" };

export default function Page() {
  return (
    <Suspense>
      <Playground />
    </Suspense>
  );
}
