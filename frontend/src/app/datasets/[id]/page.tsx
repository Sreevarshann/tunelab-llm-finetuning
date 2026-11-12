import type { Metadata } from "next";
import { Suspense } from "react";
import { DatasetDetail } from "@/components/Datasets";
import { Skeleton } from "@/components/ui";

export const metadata: Metadata = { title: "Dataset" };

async function Inner({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <DatasetDetail id={id} />;
}

export default function Page({ params }: { params: Promise<{ id: string }> }) {
  return (
    <Suspense fallback={<Skeleton className="h-64" />}>
      <Inner params={params} />
    </Suspense>
  );
}
