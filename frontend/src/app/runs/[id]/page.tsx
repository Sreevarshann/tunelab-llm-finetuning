import type { Metadata } from "next";
import { Suspense } from "react";
import { RunDetail } from "@/components/RunDetail";
import { Skeleton } from "@/components/ui";

export const metadata: Metadata = { title: "Run" };

async function Inner({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <RunDetail id={id} />;
}

export default function Page({ params }: { params: Promise<{ id: string }> }) {
  return (
    <Suspense fallback={<Skeleton className="h-96" />}>
      <Inner params={params} />
    </Suspense>
  );
}
