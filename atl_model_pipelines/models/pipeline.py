# End-to-end data pipeline with Rich console output

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
import time

from atl_model_pipelines.ingestion.ingestion_master import run_ingestion
from atl_model_pipelines.transform.transform_master import run_transforms
from atl_model_pipelines.validate.orchestrator import run_validations
from atl_model_pipelines.utils.logging import show_pipeline_table

from config import PROCESSED_DIR
import pandas as pd

console = Console()

def create_header():
    header = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║           ATL CRIME PREDICTION DATA PIPELINE                  ║
    ║     Ingestion → Transform → Validate → Save                   ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    return Panel(header, style="bold cyan", border_style="bright_cyan", expand=False)

def create_step_panel(step_num, total_steps, title, status="running"):
    if status == "running":
        emoji, style = "⏳", "bold yellow"
    elif status == "complete":
        emoji, style = "✅", "bold green"
    else:
        emoji, style = "❌", "bold red"
    return Panel(f"{emoji} [bold]{title}[/bold]", title=f"[{style}]Step {step_num}/{total_steps}[/{style}]", border_style=style, expand=False)

def create_results_table(df, out_path):
    table = Table(title="📊 Pipeline Results", box=box.ROUNDED, show_header=True, header_style="bold magenta", border_style="bright_magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    table.add_row("Output File", str(out_path.name))
    table.add_row("Full Path", str(out_path))
    table.add_row("Total Rows", f"{len(df):,}")
    table.add_row("Total Columns", str(df.shape[1]))
    table.add_row("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    if "report_date" in df.columns:
        df_temp = df.copy()
        df_temp["report_date"] = pd.to_datetime(df_temp["report_date"])
        table.add_row("Date Range", f"{df_temp['report_date'].min()} → {df_temp['report_date'].max()}")
    return table

def main():
    console.print()
    console.print(create_header())
    console.print()
    total_steps = 4
    try:
        console.print(create_step_panel(1, total_steps, "Ingestion", "running"))
        with console.status("[bold yellow]Loading raw data...", spinner="dots"):
            ingestion_output = run_ingestion()
        console.print(create_step_panel(1, total_steps, "Ingestion Complete", "complete"))
        console.print()
        
        console.print(create_step_panel(2, total_steps, "Transformations", "running"))
        with console.status("[bold yellow]Enriching features...", spinner="dots"):
            transform_output = run_transforms(ingestion_output)
            df_final = transform_output["target_crimes"]
        console.print(create_step_panel(2, total_steps, "Transformations Complete", "complete"))
        console.print()
        
        console.print(create_step_panel(3, total_steps, "Validation", "running"))
        with console.status("[bold yellow]Running quality checks...", spinner="dots"):
            df_validated = run_validations(df_final, step_name="Final target crimes", check_npu=True)
        console.print(create_step_panel(3, total_steps, "Validation Complete", "complete"))
        console.print()
        
        console.print(create_step_panel(4, total_steps, "Saving Dataset", "running"))
        out_path = PROCESSED_DIR / "target_crimes_final.parquet"
        with console.status("[bold yellow]Writing to disk...", spinner="dots"):
            df_validated.to_parquet(out_path, index=False)
            time.sleep(0.5)
        console.print(create_step_panel(4, total_steps, "Save Complete", "complete"))
        console.print()
        
        console.print(Panel("[bold green] PIPELINE COMPLETED SUCCESSFULLY [/bold green]", border_style="bright_green", expand=False))
        console.print()
        console.print(create_results_table(df_validated, out_path))
        console.print()
        console.print(Panel("[bold cyan] Pipeline Execution Summary[/bold cyan]", border_style="cyan", expand=False))
        show_pipeline_table()
        console.print()
        console.print(Panel.fit("[bold] Dataset ready for modeling! [/bold]\n\n[dim]Next steps:[/dim]\n  • Build panels: [cyan]python -m atl_model_pipelines.transform.panel_builder[/cyan]\n  • Run models: [cyan]python -m atl_model_pipelines.models.rolling_cv[/cyan]", border_style="bright_blue", title="[bold green]Success[/bold green]"))
    except Exception as e:
        console.print()
        console.print(Panel(f"[bold red] PIPELINE FAILED [/bold red]\n\n[red]Error:[/red] {str(e)}\n\n[dim]Check logs above for details.[/dim]", border_style="bright_red", title="[bold red]Error[/bold red]", expand=False))
        raise

if __name__ == "__main__":
    main()