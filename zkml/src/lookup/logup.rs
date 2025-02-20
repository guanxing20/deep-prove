//! This module contains an implementation of the [`LookupProtocol`] trait for the LogUp PIOPpub struct LogUp<E: ExtensionField>(PhantomData<E>);

use ff_ext::ExtensionField;

use serde::{Serialize, de::DeserializeOwned};

use super::LookupProtocol;

pub struct LogUp;

impl<E> LookupProtocol<E> for LogUp
where
    E::BaseField: Serialize + DeserializeOwned,
    E: ExtensionField + Serialize + DeserializeOwned,
{
}
