import click


# -----------------
# Customized click
# -----------------
def _write_dl(formatter, rows, col=12, indent=2):
    rows = [(k, click.wrap_text(v)) for k, v in rows if k is not None]
    formatter.write_dl(
        [(click.style(" " * indent + k.ljust(col), bold=True), v) for k, v in rows],
    )


class _base:
    def format_flags(self, ctx, formatter):
        formatter.write_text("Flags:")
        _write_dl(formatter, self.collect_flags(ctx))

    def collect_flags(self, ctx):
        return [
            (", ".join(param.opts), param.help)
            for param in self.get_params(ctx)
            if isinstance(param, click.Option)
        ]


class _group(_base, click.Group):
    def format_help(self, ctx, formatter):
        formatter.write_usage(ctx.command_path, "[flags] command [args]...")
        formatter.write_paragraph()
        self.format_commands(ctx, formatter)
        formatter.write_paragraph()
        self.format_custom_flags(formatter)

    def format_commands(self, ctx, formatter):
        commands = []
        for name in self.list_commands(ctx):
            cmd = self.get_command(ctx, name)
            if cmd and not cmd.hidden:
                commands.append((name, cmd))
        if commands:
            formatter.write_text("Available Commands:")
            _write_dl(
                formatter,
                [(name, cmd.get_short_help_str()) for name, cmd in commands],
            )

    def format_custom_flags(self, formatter):
        formatter.write_text("Flags:")
        custom_flags = [
            ("-h, --help", "help for voh"),
            ("--version", "Show version information"),
        ]
        _write_dl(formatter, custom_flags)


class _command(_base, click.Command):
    def format_help(self, ctx, formatter):
        self.format_usage(ctx, formatter)
        formatter.write_paragraph()
        self.format_flags(ctx, formatter)

    def format_usage(self, ctx, formatter):
        pieces = self.collect_usage_pieces(ctx)
        formatter.write_usage(
            ctx.command_path,
            " ".join(pieces).replace("[OPTIONS]", "[flags]"),
        )


# -----------------
# CLI: voh
# -----------------


@click.group(cls=_group)
@click.version_option(
    version="0.0.1",
    message="%(prog)s version is %(version)s",
)
@click.help_option("-h", "--help")
def cli():
    pass


@cli.command(cls=_command)
@click.argument("model")
@click.help_option("-h", "--help")
@click.option(
    "-f",
    "--file",
    type=str,
    metavar="string",
    help="Name of the configuration file for a model",
)
def create(model, file):
    """Create a model from a conf file"""
    from foc import cf_, lazy
    from ouch import prompt, read_conf

    from voh import dumper, size_model, voh

    conf = file and read_conf(file)
    o = voh.create(model, conf)
    o.show()
    prompt(
        "\nAre you sure to save this model?",
        ok=lazy(
            cf_(
                lambda x: dumper(
                    model=o.name,
                    path=x,
                    size=size_model(o.name),
                ),
                o.save,
            )
        ),
    )


@cli.command(cls=_command)
@click.argument("model")
@click.help_option("-h", "--help")
@click.option(
    "-f",
    "--file",
    type=str,
    metavar="string",
    help="Name of the configuration file for training",
)
def train(model, file):
    """Train a model"""

    from foc import error, lazy
    from ouch import prompt, read_conf

    from voh import default, voh

    o = voh.load(model, file and read_conf(file))
    o.show()
    print()
    o.info(default.META)
    prompt(
        "\nWant to use the above trainer to train the model?",
        fail=lazy(error, "Canceled."),
    )
    o.get_trained()


@cli.command(cls=_command)
@click.argument("model")
@click.help_option("-h", "--help")
@click.option(
    "--full",
    is_flag=True,
    show_default=True,
    default=False,
    help="Display the detailed model information in full.",
)
def show(model, full):
    """Show information for a model"""
    from voh import voh

    o = voh.load(model)
    o.info() if full else o.show()


@cli.command(cls=_command)
@click.help_option("-h", "--help")
def list():
    """List models"""
    from voh import list_models

    list_models()


@cli.command(cls=_command)
@click.argument("models", nargs=-1)
@click.help_option("-h", "--help")
def rm(models):
    """Remove a model"""
    from ouch import shell

    from voh import which_model

    for model in models:
        path = which_model(model)
        if path:
            shell(f"rm -rf {path} {path}.snap 2>/dev/null")
            print(f"deleted '{model}'")


if __name__ == "__main__":
    cli(prog_name="voh")
