library(clusterProfiler)
library(org.Hs.eg.db)
library(AnnotationDbi)
library(ggplot2)

out_dir <- "/project_antwerp/hbae/Loki_output/GO_results"
dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)

# Universe = 전체 HVG 2000 genes
universe_syms <- readLines("/project_antwerp/hbae/data/0317_hvg_2000_list.txt")
universe_mapped <- bitr(universe_syms, fromType="SYMBOL", toType="ENTREZID", OrgDb=org.Hs.eg.db)
universe_entrez <- unique(universe_mapped$ENTREZID)
cat("Universe:", length(universe_entrez), "genes\n")

run_go <- function(gene_file, label) {
  syms <- readLines(gene_file)
  mapped <- bitr(syms, fromType="SYMBOL", toType="ENTREZID", OrgDb=org.Hs.eg.db)
  entrez <- unique(mapped$ENTREZID)
  cat(label, ":", length(syms), "symbols ->", length(entrez), "ENTREZ\n")

  for (ont in c("BP", "CC", "MF")) {
    ego <- enrichGO(
      gene          = entrez,
      OrgDb         = org.Hs.eg.db,
      keyType       = "ENTREZID",
      universe      = universe_entrez,
      ont           = ont,
      pAdjustMethod = "BH",
      pvalueCutoff  = 0.05,
      qvalueCutoff  = 0.05,
      readable      = TRUE
    )

    if (!is.null(ego) && nrow(as.data.frame(ego)) > 0) {
      csv_path <- file.path(out_dir, paste0(label, "_GO_", ont, ".csv"))
      write.csv(as.data.frame(ego), csv_path, row.names=FALSE)
      cat("  Saved:", csv_path, "\n")

      p <- dotplot(ego, showCategory=20, font.size=10) +
           ggtitle(paste(label, "GO-", ont))
      ggsave(file.path(out_dir, paste0(label, "_GO_", ont, "_dotplot.png")),
             p, width=10, height=7, dpi=300)
    } else {
      cat("  No significant terms for", ont, "\n")
    }
  }
}

run_go("/project_antwerp/hbae/Loki_output/GO_high_pcc_genes.txt", "High_PCC")
run_go("/project_antwerp/hbae/Loki_output/GO_low_pcc_genes.txt",  "Low_PCC")

cat("\nDone!\n")
