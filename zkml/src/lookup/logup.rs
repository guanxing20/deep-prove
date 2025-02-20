//! This module contains an implementation of the [`LookupProtocol`] trait for the LogUp PIOPpub struct LogUp<E: ExtensionField>(PhantomData<E>);

use crate::Claim;
use anyhow::anyhow;
use ff_ext::ExtensionField;
use multilinear_extensions::mle::MultilinearExtension;
use transcript::Transcript;

use super::{
    Context, LookupProtocol, MLE, Proof, VerifierClaims,
    gkr_circuits::logup::{prove_logup, verify_logup},
};

pub struct LogUp;

impl<E: ExtensionField> LookupProtocol<E> for LogUp {}
