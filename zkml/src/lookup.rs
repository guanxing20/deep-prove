use std::marker::PhantomData;

use ff_ext::ExtensionField;
use multilinear_extensions::mle::DenseMultilinearExtension;
use transcript::Transcript;

type MLE<E> = DenseMultilinearExtension<E>;
struct Proof<E> {
    // one commitment per "column" in the lookups
    //lookups_C: Vec<(Commitment<E>,Opening,Claim)>,
    //gkr_proof: GKRProof<E>,
    //multi_C: Commitment<E>,
    _p: PhantomData<E>,
}

struct Context<E> {
    _p: PhantomData<E>,
}

trait LookupProtocol<E: ExtensionField> {
    // lookups has dimension (N,N) in case of two columns
    // with N lookup "op".
    // we must pad lookups MLEs to same dim than table
    // table can come from context
    // e.g table[i].num_vars() == lookups[i].num_vars()
    fn prove<T: Transcript<E>>(table: Vec<MLE<E>>, lookups: Vec<MLE<E>>,t: &mut T) -> anyhow::Result<Proof<E>>;
    
    // commitments to the lookups, one commitment per "column"
    fn verify<T: Transcript<E>>(ctx: Context<E>, proof: Proof<E>, t: &mut T) -> anyhow::Result<()>;
    
}