"""The proxy builder: prepare a policy once, load records against it many times.

The developer-facing lifecycle from the design doc. prepare() resolves the
governance machinery once, before anything runs. load() is where the guarantee
starts: it pins the schema, and every call returns a new bound governed frame,
so a second load never mutates the first.
"""

from .admission import admit


class PreparedPolicy:
    """The product of prepare(): a proxy factory bound to one compiled policy."""

    def __init__(self, policy, tn_governance, evaluator, dtypes=None,
                 label_map=None):
        self.policy = policy
        self.tn_governance = dict(tn_governance)
        self.evaluator = evaluator
        self.dtypes = dict(dtypes) if dtypes else None
        self.label_map = dict(label_map) if label_map else None

    def load(self, frame):
        """Admit one frame and return a new governed proxy bound to it."""
        return admit(frame, self.policy, self.tn_governance, self.evaluator,
                     dtypes=self.dtypes, label_map=self.label_map)


class GovernedPolars:
    """Entry point matching the design doc: GovernedPolars.prepare(...).load(...)."""

    @staticmethod
    def prepare(policy, tn_governance=None, evaluator=None, *, dtypes=None):
        """Build a PreparedPolicy from parts, or from one registry entry.

        The one-argument form is the design doc's lifecycle: the argument is
        what ``registry.get(name)`` returned, and it carries the compiled
        policy, the TN governance body, the declared dtypes, and the OPA
        artifacts to build the evaluator from.
        """
        if tn_governance is None and evaluator is None:
            entry = policy
            return PreparedPolicy(entry.policy, entry.tn_governance,
                                  entry.evaluator(), dtypes=entry.dtypes,
                                  label_map=entry.label_map)
        return PreparedPolicy(policy, tn_governance, evaluator, dtypes=dtypes)

    @staticmethod
    def load_from_tn(wire, *, session, use, selection, policies, decide,
                     group="finance", rows_field="rows"):
        """Admit exact TN bytes for the declared use before constructing a frame."""
        from .tn_boundary import load_from_tn
        return load_from_tn(wire, session=session, use=use, selection=selection,
                            policies=policies, decide=decide, group=group,
                            rows_field=rows_field)
