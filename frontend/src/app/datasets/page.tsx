import type { Metadata } from "next";
import { DatasetsPage } from "@/components/Datasets";

export const metadata: Metadata = { title: "Datasets" };

export default function Page() {
  return <DatasetsPage />;
}
