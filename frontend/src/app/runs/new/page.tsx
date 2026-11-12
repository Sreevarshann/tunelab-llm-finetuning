import type { Metadata } from "next";
import { Suspense } from "react";
import { NewRun } from "@/components/NewRun";

export const metadata: Metadata = { title: "New run" };

export default function Page() {
  return (
    <Suspense>
      <NewRun />
    </Suspense>
  );
}
