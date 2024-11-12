import click


class vohGroup(click.Group):
    def format_help(self, ctx, formatter):
        formatter.write_usage(ctx.command_path, "[FLAGS] COMMAND [ARGS]...")
        formatter.write_paragraph()

        self.format_commands(ctx, formatter)

        formatter.write_paragraph()
        formatter.write_text("Flags:")
        custom_flags = [
            ("-h, --help", "help for voh"),
            ("--version", "Show version information"),
        ]
        self.write_dl(formatter, custom_flags, col=12)

    def format_commands(self, ctx, formatter):
        commands = []
        for subcommand in self.list_commands(ctx):
            cmd = self.get_command(ctx, subcommand)
            if cmd is None or cmd.hidden:
                continue
            commands.append((subcommand, cmd))

        if commands:
            formatter.write_text("Available Commands:")
            self.write_dl(
                formatter,
                [(name, cmd.get_short_help_str()) for name, cmd in commands],
                col=12,
            )

    def write_dl(self, formatter, rows, col):
        rows = [(k, click.wrap_text(v)) for k, v in rows if k is not None]
        formatter.write_dl(
            [(click.style(k.ljust(col), bold=True), v) for k, v in rows],
        )


@click.group(cls=vohGroup)
@click.help_option("-h", "--help")
@click.version_option(version="0.0.1", message="%(prog)s version %(version)s")
def cli():
    pass


@cli.command()
@click.argument("model")
@click.option("-f", "--file", type=str, help="Name of the conf file")
def create(model, file):
    """Create a model from a conf file"""
    click.echo(f"Creating model {model} with config file {file}")


@cli.command()
@click.argument("model")
def show(model):
    """Show information for a model"""
    click.echo(f"Showing information for model {model}")


@cli.command()
def list():
    """List models"""
    click.echo("Listing all models")


@cli.command()
@click.argument("model")
def train(model):
    """Train a model"""
    click.echo(f"Training model {model}")


@cli.command()
@click.argument("model")
def rm(model):
    """Remove a model"""
    click.echo(f"Removing model {model}")


if __name__ == "__main__":
    cli(prog_name="voh")
