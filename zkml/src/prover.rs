use ff_ext::ExtensionField;

struct Prover<E> {
    input: Vec<E>,
    output: Vec<E>,
    model: Vec<E>,
}

impl<E> Prover<E>
where
    E: ExtensionField,
{
    pub fn new(model: Vec<E>, input: Vec<E>) {}
}

