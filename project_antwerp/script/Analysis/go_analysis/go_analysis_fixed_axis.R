library(clusterProfiler)
library(org.Hs.eg.db)
library(ggplot2)

out_dir <- "/project_antwerp/hbae/Loki_output/GO_results_fixed"
dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)

universe_syms <- readLines("/project_antwerp/hbae/data/0317_hvg_2000_list.txt")
universe_mapped <- bitr(universe_syms, fromType="SYMBOL", toType="ENTREZID", OrgDb=org.Hs.eg.db)
universe_entrez <- unique(universe_mapped$ENTREZID)

# 먼저 두 그룹 모두 ego 객체 생성
get_ego <- function(gene_file, ont) {
  syms <- readLines(gene_file)
  mapped <- bitr(syms, fromType="SYMBOL", toType="ENTREZID", OrgDb=org.Hs.eg.db)
  entrez <- unique(mapped$ENTREZID)
  enrichGO(
    gene=entrez, OrgDb=org.Hs.eg.db, keyType="ENTREZID",
    universe=universe_entrez, ont=ont,
    pAdjustMethod="BH", pvalueCutoff=0.05, qvalueCutoff=0.05, readable=TRUE
  )
}

for (ont in c("BP", "CC", "MF")) {
  ego_high <- get_ego("/project_antwerp/hbae/Loki_output/GO_high_pcc_genes.txt", ont)
  ego_low  <- get_ego("/project_antwerp/hbae/Loki_output/GO_low_pcc_genes.txt",  ont)

  # 두 그룹 GeneRatio 최대값으로 x축 범위 통일
  get_max_ratio <- function(ego) {
    if (is.null(ego) || nrow(as.data.frame(ego)) == 0) return(0)
    df <- as.data.frame(ego)
    max(sapply(df$GeneRatio, function(x) eval(parse(text=x))))
  }
  xmax <- max(get_max_ratio(ego_high), get_max_ratio(ego_low)) * 1.1

  for (label in c("High_PCC", "Low_PCC")) {
    ego <- if (label == "High_PCC") ego_high else ego_low
    if (is.null(ego) || nrow(as.data.frame(ego)) == 0) next

    p <- dotplot(ego, showCategory=15, font.size=10) +
         ggtitle(paste(label, "GO -", ont)) +
         xlim(0, xmax) +
         theme_bw(base_size=11) +
         theme(axis.text.y=element_text(size=9))

    ggsave(file.path(out_dir, paste0(label, "_GO_", ont, "_fixed.png")),
           p, width=9, height=7, dpi=300)
    cat("Saved:", label, ont, "\n")
  }
}
cat("Done!\n")
